import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, hid_size, out_int, out_slot, dropout=0.):
        super(JointBERT, self).__init__(config)

        # Base bert model
        self.bert = BertModel(config)

        # Fine tune for slot and intent
        self.slot_classifier = nn.Linear(hid_size, out_slot)
        self.int_classifier = nn.Linear(hid_size, out_int)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = bert_output.last_hidden_state     # Last hidden state (slot)
        pooled_output = bert_output.pooler_output   # Pooler output (intent)

        # Apply dropout (if specified)
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        slots = self.slot_classifier(sequence_output)
        intent = self.int_classifier(pooled_output)

        slots = slots.permute(0, 2, 1)

        return slots, intent