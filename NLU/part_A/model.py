import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS, self).__init__()
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)
        
        # Double hid_size because of bidirectionality
        self.slot_out = nn.Linear(hid_size * 2, out_slot) 
        self.intent_out = nn.Linear(hid_size * 2, out_int)

        # Dropout layer
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance)
        
        # Apply dropout after embedding layer
        utt_emb = self.dropout(utt_emb)
        
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
       
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        
        # Concatenate bidirectional hidden states (both directions)
        last_hidden = torch.cat([last_hidden[-2], last_hidden[-1]], dim=1)
        
        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)
        
        slots = slots.permute(0,2,1)
        
        return slots, intent