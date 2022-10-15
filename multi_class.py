import pickle
import re

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from transformers import BertModel, BertTokenizer, AutoTokenizer

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from utils.preprocessor import Preprocessor
from model.classifier import Classifier
from utils.trainer import MultiClassTrainer

from transformers import BertConfig, BertTokenizer

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import pandas as pd

if __name__ == '__main__':
    dm = Preprocessor(batch_size=100)
    
    config = BertConfig()
    model = Classifier(bert_config = config, num_classes = 5, dropout = 0.5).from_pretrained('indolem/indobert-base-uncased')
    

    # 1e-4 = acc = 0.9 loss 0.1

    model = MultiClassTrainer(
        lr = 1e-5,
        bert_config = config,
        dropout=0.3
    )

    logger = TensorBoardLogger("logs", name="bert_classifier")

    trainer = pl.Trainer(gpus = 1,
                         max_epochs = 20, 
                         default_root_dir = "./checkpoints/class",
                         logger = logger)
    trainer.fit(model, datamodule = dm)

    trainer.predict(model = model, datamodule = dm)
