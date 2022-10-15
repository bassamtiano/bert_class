import pickle
from random import shuffle
import re
import os
import sys

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import pytorch_lightning as pl

from transformers import BertTokenizer

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import pandas as pd
from tqdm import tqdm

class Preprocessor(pl.LightningDataModule):

    def __init__(self, 
                 max_length = 100, 
                 batch_size = 50,
                 recreate = False):
        
        super(Preprocessor, self).__init__()
        self.label2id = {
            'bola': 0, 
            'news': 1,
            'bisnis':2, 
            'tekno':3, 
            'otomotif':4
        }

        self.max_length = max_length
        self.batch_size = batch_size

        self.recreate = recreate

        self.preprocessed_dir = 'data/preprocessed/'

        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        self.tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

    def load_data(self):
        with open('data/training.res', 'rb') as tpr:
            train_pkl = pickle.load(tpr)
            train = pd.DataFrame({'title': train_pkl[0], 'label': train_pkl[1]})

        with open('data/testing.res', 'rb') as tspr:
            test_pkl = pickle.load(tspr)
            test = pd.DataFrame({'title': test_pkl[0], 'label': test_pkl[1]})
        
        # konversi label karakter (bola) ke label angka (0)
        train.label = train.label.map(self.label2id)
        test.label = test.label.map(self.label2id)

        return train, test

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

        return self.stemmer.stem(string)

    def arrange_data(self, datas, preprocessed_dir, type):
        x_input_ids, x_token_type_ids, x_attention_mask, y = [], [], [], []

        for i, tr_d in enumerate(tqdm(datas.values.tolist())):
            title = self.clean_str(tr_d[0])
            label = tr_d[1]
            
            binary_lbl = [0] * 5
            binary_lbl[label] = 1
            
            tkn = self.tokenizer(text = title, 
                                 max_length= self.max_length, 
                                 padding='max_length',
                                 truncation=True)
            
            
            x_input_ids.append(tkn['input_ids'])
            x_token_type_ids.append(tkn['token_type_ids'])
            x_attention_mask.append(tkn['attention_mask'])
            y.append(binary_lbl)

            # if i > 100: break

        
        x_input_ids = torch.tensor(x_input_ids)
        x_token_type_ids = torch.tensor(x_token_type_ids)
        x_attention_mask = torch.tensor(x_attention_mask)
        y = torch.tensor(y)

        tensor_dataset = TensorDataset(x_input_ids, x_token_type_ids, x_attention_mask, y)

        if type == 'train':
            
            train_tensor_dataset, valid_tensor_dataset = torch.utils.data.random_split(tensor_dataset, [round(len(x_input_ids) * 0.8), round(len(x_input_ids) * 0.2)])
            torch.save(train_tensor_dataset, f"{preprocessed_dir}/train.pt")
            torch.save(valid_tensor_dataset, f"{preprocessed_dir}/valid.pt")

            return train_tensor_dataset, valid_tensor_dataset
        else:
            
            torch.save(tensor_dataset, f"{preprocessed_dir}/test.pt")
            return tensor_dataset

    def preprocessing(self):

        train, test = self.load_data()

        if not os.path.exists(f"{self.preprocessed_dir}/train.pt") or not os.path.exists(f"{self.preprocessed_dir}/valid.pt") or self.recreate:
            print("Creating Train and Validation dataset")
            train_data, valid_data = self.arrange_data(train, 
                                                       preprocessed_dir = self.preprocessed_dir, 
                                                       type="train")
        else:
            print("Loading Train and Validation dataset")
            train_data = torch.load(f"{self.preprocessed_dir}/train.pt")
            valid_data = torch.load(f"{self.preprocessed_dir}/valid.pt")

        if not os.path.exists(f"{self.preprocessed_dir}/test.pt") or self.recreate:
            print("Creating Test Dataset")
            test_data = self.arrange_data(test, 
                                          preprocessed_dir = self.preprocessed_dir, 
                                          type="test")
        else:
            print("Loading Test and Validation dataset")
            test_data = torch.load(f"{self.preprocessed_dir}/test.pt")

        # train_loader = DataLoader(train_data, shuffle = True, batch_size = self.batch_size)
        # valid_loader = DataLoader(valid_data, shuffle = True, batch_size = self.batch_size)
        # test_loader = DataLoader(test_data, shuffle = False, batch_size = self.batch_size)

        # return train_loader, valid_loader, test_loader
        return train_data, valid_data, test_data
    
    def setup(self, stage = None):
        train_data, valid_data, test_data = self.preprocessing()
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
        


        
