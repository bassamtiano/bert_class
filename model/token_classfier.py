import torch
import torch.nn as nn

from transformers import BertModel

class TokenClassifier(nn.Module):
    def __init__(self, n_vocab) -> None:
        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased')
        self.lf = nn.Linear(768, n_vocab)

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.bert(input_ids = input_ids,
                           token_type_ids = token_type_ids,
                           attention_mask = attention_mask)

        