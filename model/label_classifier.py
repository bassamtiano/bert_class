import torch
import torch.nn as nn

from transformers import BertModel

class LabelClassifier(nn.Module):
    def __init__(self, n_out, dropout) -> None:
        super(LabelClassifier, self).__init__().__init__()

        self.l1 = BertModel.from_pretrained('indolem/indobert-base-uncased')
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, n_out)
        

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_out = self.l1(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        hidden_state = bert_out[0]

        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output