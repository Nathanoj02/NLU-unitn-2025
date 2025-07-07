from functools import partial
from torch.utils.data import DataLoader

from functions import *
from utils import *
from model import LM_LSTM_weight_tying, LM_LSTM_var_dropout, LM_GRU_var_dropout

import torch.optim as optim
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

    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=DEVICE),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=DEVICE))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=DEVICE))

    # Use same size for hidden and embedding (only emb_size will be passed) for weight tying
    hid_size = 400
    emb_size = 400

    # Learning rate for NT-AvSGD (also SGD)
    lr = 0.5
    clip = 5

    # Dropout and number of layers
    emb_dropout = 0.1
    out_dropout = 0.1
    n_layers = 1

    vocab_len = len(lang.word2id)

    # Model instantiation
    model = LM_LSTM_var_dropout(emb_size, vocab_len, pad_index=lang.word2id["<pad>"],
                   emb_dropout = emb_dropout, out_dropout = out_dropout, n_layers = n_layers).to(DEVICE)
    model.apply(init_weights)

    optimizer = NT_AvSGD(model.parameters(), lr=lr, L=len(train_loader), n=5)

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

            # Only for NT-AvSGD - update validation loss (for triggering)
            optimizer.update_val_loss(ppl_dev)
            # --------------------------------

            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')

                # Save the model
                # path = 'bin/LSTM_var_dropout.pt'
                # torch.save(model.state_dict(), path)

                patience = 3
            else:
                patience -= 1
                
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    # Only for NT-AvSGD - average weights after traning (last line in the paper's pseudo-code)
    optimizer.apply_average()
    # -----------------------

    best_model.to(DEVICE)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)
