import json
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
import torch.utils.data as data

PAD_TOKEN = 0


def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


def split_set(tmp_train_raw, portion=0.1):
    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    return train_raw, dev_raw


class Lang():
    def __init__(self, intents, slots, tokenizer):
        # No words required for BERT

        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}

        self.tokenizer = tokenizer  # Added tokenizer

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
    

class IntentsAndSlots (data.Dataset):
    def __init__(self, dataset, lang, tokenizer, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.tokenizer = tokenizer  # Added tokenizer
        self.unk = unk

        self.mask = []
        self.input_ids = []

        for x in dataset:
            self.intents.append(x['intent'])

            utt = x['utterance'].replace("'",'') # Remove the ' as it gives problems later
            tok_utt = tokenizer(utt)
            ids = tok_utt['input_ids'] # tok_utt has 'attention_mask', 'input_ids' and 'token_type_ids'
            tokens = tokenizer.convert_ids_to_tokens(ids)

            slots = x['slots'].split()  # Split slots to get a list
            word_counter = 0    # Used to keep track of the number of words -> to get the correct slot later

            # Handling of the subtokenization issue: subtokens (the ones that start with ##) are transformed into PAD
            # and will be ignored later by BERT
            slots_tmp = []
            for token in tokens:
                if token.startswith('##') or token == '.' or token in [tokenizer.cls_token, tokenizer.sep_token]:
                    slots_tmp.append(lang.slot2id['pad'])
                else :
                    slot = slots[word_counter]
                    word_counter += 1
                    slots_tmp.append(lang.slot2id[slot])

            input_ids_tmp = [id for id in ids]

            self.slots.append(slots_tmp)
            self.input_ids.append(input_ids_tmp)

        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

        for x in self.input_ids:
            self.mask.append([1 if el != PAD_TOKEN else 0 for el in x])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = torch.Tensor(self.input_ids[idx])
        slots = torch.Tensor(self.slots[idx])
        intent = self.intent_ids[idx]
        mask = torch.Tensor(self.mask[idx])
        sample = {'input_ids': input_ids, 'slots': slots, 'intent': intent, 'mask': mask}
        return sample

    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data, mapper):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res


def collate_fn(data, device):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths
    
    data.sort(key=lambda x: len(x['input_ids']), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    # Changed this part to match the new item (using input_ids and mask)
    src_input_ids, _ = merge(new_item['input_ids'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])

    mask, _ = merge(new_item['mask'])

    src_input_ids = src_input_ids.to(device)
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    mask = mask.to(device)

    new_item["input_ids"] = src_input_ids
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    new_item["mask"] = mask
    return new_item