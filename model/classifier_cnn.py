import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import BertModel

class ClassifierCNN(pl.LightningModule):
    def __init__(self,
                 labels,
                 lr = 1e-4,
                 embedding_dim = 768,
                 in_channels = 8, 
                 out_channels = 32,
                 num_classes = 12,
                 kernel_size = 10,
                 dropout = 0.3) -> None:

        super(ClassifierCNN).__init__()

        ks = 3

        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased')

        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, embedding_dim), padding=(2, 0), groups=4)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (4, embedding_dim), padding=(3, 0), groups=4)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (5, embedding_dim), padding=(4, 0), groups=4)

        self.l1 = nn.Linear(ks * out_channels, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_out = self.bert(input_ids = input_ids, 
                             attention_mask = attention_mask, 
                             token_type_ids = token_type_ids)

        hidden_state = bert_out[2]

        hidden_state = torch.stack(hidden_state, dim = 1)
        hidden_state = hidden_state[:, -8:]

        x = [
            F.relu(self.conv1(hidden_state).squeeze(3)),
            F.relu(self.conv2(hidden_state).squeeze(3)),
            F.relu(self.conv3(hidden_state).squeeze(3))
        ]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        x = self.dropout(x)
        logit = self.l1(x)
        logit = self.sigmoid(logit)
        return logit

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer
        
    def training_step(self, batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        out = self(input_ids = x_input_ids,
                   attention_mask = x_attention_mask,
                   token_type_ids = x_token_type_ids)

        loss = self.criterion(out, y.float())
        self.log("train_loss", loss)

        return {"loss": loss, "predictions": out, "labels": y}

