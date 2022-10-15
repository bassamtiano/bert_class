import sys
import random

import torch
import torch.nn as nn

import pytorch_lightning as pl

from sklearn.metrics import classification_report
from model.classifier import Classifier
from model.label_classifier import LabelClassifier



class MultiClassTrainer(pl.LightningModule):
    def __init__(self, lr, bert_config, dropout) -> None:
        super().__init__()
        self.lr = lr

        torch.manual_seed(1)
        random.seed(43)

        # self.model = Classifier(bert_config = bert_config, num_classes = 5).from_pretrained('indolem/indobert-base-uncased')
        self.model = LabelClassifier(n_out=5, dropout=dropout)
        self.criterion = nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = train_batch

        out = self.model(input_ids = x_input_ids, 
                         token_type_ids = x_token_type_ids, 
                         attention_mask = x_attention_mask)
        # OUT = PREDICTION
        # TARGET = GRAND TRUTH
        loss = self.criterion(out, target = y.float())

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()
        report = classification_report(true, pred, output_dict = True, zero_division = 0)
        self.log("accuracy", report['accuracy'], prog_bar = True)
        self.log("loss", loss)
        
        return loss
    
    def validation_step(self, valid_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = valid_batch
    
        out = self.model(input_ids = x_input_ids, 
                         token_type_ids = x_token_type_ids, 
                         attention_mask = x_attention_mask)

        loss = self.criterion(out, target = y.float())
                
        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()
        report = classification_report(true, pred, output_dict = True, zero_division = 0)
        self.log("accuracy", report['accuracy'], prog_bar = True)
        self.log("loss", loss)

        return loss

    def predict_step(self, test_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = test_batch
    
        out = self.model(input_ids = x_input_ids, 
                         token_type_ids = x_token_type_ids, 
                         attention_mask = x_attention_mask)
        
        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()

        return pred
        report = classification_report(true, pred, output_dict = True, zero_division = 0)
        # self.log("accuracy", report['accuracy'], prog_bar = True)
