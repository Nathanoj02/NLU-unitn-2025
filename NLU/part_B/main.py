from functions import *
from utils import load_data, IntentsAndSlots, split_set, Lang, collate_fn, PAD_TOKEN
from model import JointBERT

from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from transformers import BertConfig, BertTokenizer
from functools import partial
import os

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tmp_train_raw = load_data(os.path.join('dataset','train.json'))
    test_raw = load_data(os.path.join('dataset','test.json'))
    
    train_raw, dev_raw = split_set(tmp_train_raw, portion=0.1)

    # Download the tokenizer
    bert_model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(intents, slots, tokenizer)

    train_dataset = IntentsAndSlots(train_raw, lang, tokenizer)
    dev_dataset = IntentsAndSlots(dev_raw, lang, tokenizer)
    test_dataset = IntentsAndSlots(test_raw, lang, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=partial(collate_fn, device=device), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=partial(collate_fn, device=device))
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=partial(collate_fn, device=device))
    
    # Get the configuration for BERT
    config = BertConfig.from_pretrained(bert_model_name)
    hid_size = config.hidden_size   # BERT hidden size is 768, but this is a better way to get it

    lr = 1e-4
    clip = 5

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)

    # Model instantiation
    model = JointBERT(config=config, hid_size=hid_size, out_int=out_int, out_slot=out_slot, dropout=0.1).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    n_epochs = 200
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    for x in tqdm(range(1,n_epochs)):
        loss = train_loop(train_loader, optimizer, criterion_slots,
                        criterion_intents, model, clip=clip)
        if x % 5 == 0:
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots,
                                                        criterion_intents, model, lang)
            losses_dev.append(np.asarray(loss_dev).mean())

            f1 = results_dev['total']['f']
            
            if f1 > best_f1:
                best_f1 = f1
                
                # Save the model
                path = 'bin/jointBERT.pt'
                torch.save(model.state_dict(), path)

                patience = 3
            else:
                patience -= 1
            if patience <= 0:
                break

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots,
                                            criterion_intents, model, lang)
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])

    # Plot loss
    # plot_loss(sampled_epochs, losses_train, losses_dev)
