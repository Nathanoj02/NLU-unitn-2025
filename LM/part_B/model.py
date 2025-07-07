import torch.nn as nn
from torch.autograd import Variable


class LM_LSTM_weight_tying(nn.Module):
    def __init__(self, emb_size, output_size, pad_index=0, n_layers=1):
        super(LM_LSTM_weight_tying, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, emb_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.output = nn.Linear(emb_size, output_size)

        # Weight tying between embedding layer and last linear layer
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _  = self.lstm(emb)
        output = self.output(lstm_out).permute(0,2,1)
        return output
    

# Variational dropout module - if called in init it generates a constant mask through all LSTM's time steps
class VariationalDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        # Do not apply dropout when evaluating
        if not self.training or not dropout:
            return x
        
        # Build dropout mask
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x
    
class LM_LSTM_var_dropout(nn.Module):
    def __init__(self, emb_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_var_dropout, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, emb_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.output = nn.Linear(emb_size, output_size)
        
        # Weight tying
        self.output.weight = self.embedding.weight
        
        # Variational dropout
        self.dropout1 = VariationalDropout()
        self.dropout2 = VariationalDropout()
        self.emb_dropout = emb_dropout
        self.out_dropout = out_dropout
    
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb_dropout = self.dropout1(emb, dropout=self.emb_dropout)
        lstm_out, _  = self.lstm(emb_dropout)
        out_dropout = self.dropout2(lstm_out, dropout=self.out_dropout)
        output = self.output(out_dropout).permute(0,2,1)
        return output
    

class LM_GRU_var_dropout(nn.Module):
    def __init__(self, emb_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_GRU_var_dropout, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        # Pytorch's GRU layer: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
        self.gru = nn.GRU(emb_size, emb_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.output = nn.Linear(emb_size, output_size)
        
        # Weight tying
        self.output.weight = self.embedding.weight
        
        # Variational dropout
        self.dropout1 = VariationalDropout()
        self.dropout2 = VariationalDropout()
        self.emb_dropout = emb_dropout
        self.out_dropout = out_dropout
    
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb_dropout = self.dropout1(emb, dropout=self.emb_dropout)
        gru_out, _  = self.gru(emb_dropout)
        out_dropout = self.dropout2(gru_out, dropout=self.out_dropout)
        output = self.output(out_dropout).permute(0,2,1)
        return output