import sys
import random

import torch
import torch.nn as nn

import pytorch_lightning as pl

from sklearn.metrics import classification_report
from model.cnn_bert import CNNBert

from torchmetrics import AUROC

class TrainerMultilabel(pl.LightningModule):
    def __init__(self,  
                 labels, 
                 pretrained_model = None, 
                 lr = 1e-4
                 ) -> None:
        super(TrainerMultilabel, self).__init__()

        self.lr = lr
        self.labels = labels

        torch.manual_seed(1)
        random.seed(43)

        self.model = CNNBert()

        if pretrained_model:
            self.model.from_pretrained(pretrained_model)
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.BCELoss()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = train_batch

        out = self.model(input_ids = x_input_ids,
                         token_type_ids = x_token_type_ids,
                         attention_mask = x_attention_mask)

        loss = self.criterion(out, y.float())

        self.log("train_loss", loss)
        
        return {"loss": loss, "predictions": out, "labels": y}
        
    def validation_step(self, valid_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = valid_batch

        out = self.model(input_ids = x_input_ids,
                         token_type_ids = x_token_type_ids,
                         attention_mask = x_attention_mask)
        
        loss = self.criterion(out.cpu(), y.float().cpu())

        self.log("val_loss", loss)

        return loss

    def predict_step(self, test_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = test_batch
        out = self.model(input_ids = x_input_ids,
                         token_type_ids = x_token_type_ids,
                         attention_mask = x_attention_mask)
        
        loss = self.criterion(out.cpu(), y.float().cpu())

        return out


    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        # results = []

        for i, name in enumerate(self.labels):
            auroc = AUROC(num_classes=len(self.labels))
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            # results.append(class_roc_auc)
            print(f"{name} \t: {class_roc_auc}")

            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)

        # print(results)
