import sys
import torch
import torch.nn as nn

from transformers import BertModel

class CNNBert(nn.Module):
    def __init__(self, 
                 in_channels = 768, 
                 out_channels = 12,
                 kernel_size = 10) -> None:
        super().__init__()

        self.bert_model = BertModel.from_pretrained('indolem/indobert-base-uncased')
        self.conv = nn.Conv2d(in_channels = in_channels, 
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              stride = 1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_out = self.bert_model(input_ids = input_ids, 
                                   attention_mask = attention_mask, 
                                   token_type_ids = token_type_ids)
        hidden_state = bert_out[0]
        test = self.conv(hidden_state)
        print(test)
        sys.exit()