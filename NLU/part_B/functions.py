import torch
import torch.nn as nn

from conll import evaluate
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from transformers import BertModel

from utils import PAD_TOKEN


def init_weights(mat):
    for m in mat.modules():
        if isinstance(m, (BertModel,)):
                continue
        # Initialize only linear layers in joint bert model
        else:
            if type(m) in [nn.Linear]:
                nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()

        slots, intent = model(sample['input_ids'], sample['mask'])  # Call to model forward

        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    
    with torch.no_grad():
        for sample in data:

            slots, intents = model(sample['input_ids'], sample['mask']) # Call to model forward

            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            
            out_intents = [lang.id2intent[x]
                           for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['input_ids'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]

                # Use of tokenizer to convert to tokens (instead of lang)
                utterance = [lang.tokenizer.convert_ids_to_tokens(el) for el in utt_ids]

                to_decode = seq[:length].tolist()

                # Remove pad slots
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots) if elem != 'pad'])

                tmp_seq = []
                for id_el, el in enumerate(to_decode):
                    if gt_ids[id_el] != PAD_TOKEN : # Skip pad tokens
                        tmp_seq.append((utterance[id_el], lang.id2slot[el]))

                hyp_slots.append(tmp_seq)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}

    report_intent = classification_report(ref_intents, hyp_intents,
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array


def plot_loss(sampled_epochs, losses_train, losses_dev):
    plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
    plt.title('Train and Dev Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(sampled_epochs, losses_train, label='Train loss')
    plt.plot(sampled_epochs, losses_dev, label='Dev loss')
    plt.legend()
    plt.show()