import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel

class Classifier(BertPreTrainedModel):
    def __init__(self, 
                 bert_config,
                 dropout = 0.3,
                 num_classes = 5):
        super(Classifier ,self).__init__(bert_config)
        self.bert = BertModel(bert_config) 
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        # self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features = bert_config.hidden_size, 
                                    out_features = num_classes)
        # self.relu = nn.ReLU()
        self.unfreeze_bert_encoder()

    def freeze_bert_encoder(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def unfreeze_bert_encoder(self):
        for p in self.bert.parameters():
            p.requires_grad = True

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.bert(input_ids,
                                  token_type_ids,
                                  attention_mask)
        pooled_output = self.dropout(output[1])
        logits = self.classifier(pooled_output)
        # out = self.relu(logits)
        return logits