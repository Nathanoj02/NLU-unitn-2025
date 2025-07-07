from functools import partial
from torch.utils.data import DataLoader

from functions import *
from utils import *
from model import LM_RNN, LM_LSTM, LM_LSTM_dropout
import torch.optim as optim

import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import numpy as np
import torch.nn as nn

if __name__ == "__main__":
    train_raw = read_file("dataset/ptb.train.txt")
    dev_raw = read_file("dataset/ptb.valid.txt")
    test_raw = read_file("dataset/ptb.test.txt")

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=DEVICE),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=DEVICE))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=DEVICE))

    hid_size = 200
    emb_size = 300

    lr_sgd = 0.5    # For SGD
    lr_adam = 1e-3  # For AdamW
    clip = 5

    # Dropout and number of layers
    emb_dropout = 0.1
    out_dropout = 0.1
    n_layers = 1

    vocab_len = len(lang.word2id)

    # Model instantiation
    model = LM_LSTM_dropout(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"],
                   emb_dropout = emb_dropout, out_dropout = out_dropout, n_layers = n_layers).to(DEVICE)
    
    model.apply(init_weights)

    # optimizer = optim.SGD(model.parameters(), lr=lr_sgd)
    optimizer = optim.AdamW(model.parameters(), lr=lr_adam)

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)

            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')

                # Save the model
                # path = 'bin/LSTM_dropout.pt'
                # torch.save(model.state_dict(), path)

                patience = 3
            else:
                patience -= 1
                
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    best_model.to(DEVICE)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)
