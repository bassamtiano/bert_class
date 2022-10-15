import sys
import pickle
import re

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import pandas as pd

import pytorch_lightning as pl

from transformers import BertTokenizer

class PreprocessorToxic(pl.LightningDataModule):

    def __init__(self):
        self.tokenizers = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')
        self.max_length = 100

    def clean_str(self, string):
        string = string.lower()
        string = re.sub(r"[^A-Za-z0-9(),!?\'\-`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\n", "", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = string.strip()

        return string

    def load_data(self):
        data = pd.read_csv('data/preprocessed_indonesian_toxic_tweet.csv')

        # Semua kolom di cek yang kosong di drop
        data = data.dropna(how="any")
        # Spesifik remove karakter null berdasarkan kolom
        data = data[data[['Tweet', 'HS_Strong']].notna()]
        
        tweet = data["Tweet"].apply(lambda x: self.clean_str(x))
        tweet = tweet.values.tolist()
        label = data.drop(["Tweet"], axis = 1)
        label = label.values.tolist()

        x_input_ids, x_token_type_ids, x_attention_mask, y = [], [], [], []

        for tw in tweet:
            tkn_tweet = self.tokenizers(text = tw,
                                        max_length = self.max_length,
                                        padding = 'max_length',
                                        truncation = True)
            
            x_input_ids.append(tkn_tweet['input_ids'])
            x_token_type_ids.append(tkn_tweet['token_type_ids'])
            x_attention_mask.append(tkn_tweet['attention_mask'])

        x_input_ids = torch.tensor(x_input_ids)
        x_token_type_ids = torch.tensor(x_token_type_ids)
        x_attention_mask = torch.tensor(x_attention_mask)
        y = torch.tensor(label)

        print(y.shape)
        sys.exit()

        tensor_dataset = TensorDataset(x_input_ids, x_token_type_ids, x_attention_mask, y)
        # Standard
        # 80% (Training validation) 20% (testing)
        # training = 90% validation = 10%

        train_valid_dataset, test_dataset = torch.utils.data.random_split(
            tensor_dataset, [
                round(len(tensor_dataset) * 0.8), 
                round(len(tensor_dataset) * 0.2)
            ]
        )

        train_len = round(len(train_valid_dataset) * 0.9)
        valid_len = len(train_valid_dataset) - round(len(train_valid_dataset) * 0.9)

        train_dataset, valid_dataset = torch.utils.data.random_split(
            train_valid_dataset,
            [train_len, valid_len]
        )

        return train_dataset, valid_dataset, test_dataset

    def setup(self, stage = None):
        train_data, valid_data, test_data = self.load_data()
        if stage == "fit":
            self.train_data = train_data
            self.valid_data = valid_data
        elif stage == "predict":
            self.test_data = test_data

    def train_dataloader(self):
        sampler = RandomSampler(self.train_data)
        return DataLoader(
            dataset = self.train_data,
            batch_size = self.batch_size,
            sampler = sampler,
            num_workers = 1
        )

    def val_dataloader(self):
        sampler = RandomSampler(self.valid_data)
        return DataLoader(
            dataset = self.valid_data,
            batch_size = self.batch_size,
            sampler = sampler,
            num_workers = 1
        )

    def predict_dataloader(self):
        sampler = SequentialSampler(self.test_data)
        return DataLoader(
            dataset = self.test_data,
            batch_size = self.batch_size,
            sampler = sampler,
            num_workers = 1
        )

if __name__ == '__main__':
    pretox = PreprocessorToxic()
    pretox.load_data()