import sys

import torch
import torch.nn as nn

import pytorch_lightning as pl
from utils.combined_trainer_multilabel import CombinedTrainerMultilabel



from transformers import BertTokenizer

if __name__ == '__main__':

    labels = [
        'HS', 
        'Abusive', 
        'HS_Individual', 
        'HS_Group', 
        'HS_Religion', 
        'HS_Race', 
        'HS_Physical', 
        'HS_Gender', 
        'HS_Other', 
        'HS_Weak', 
        'HS_Moderate', 
        'HS_Strong'
    ]

    model = CombinedTrainerMultilabel(labels)

    trainer = pl.Trainer(gpus = 1,
                         max_epochs = 10,
                         default_root_dir = "./checkpoints/labels")


    
    trained_model = CombinedTrainerMultilabel.load_from_checkpoint(
        "./logs/bert_multilabel/version_1/checkpoints/epoch=6-step=924.ckpt",
        labels = labels
    )

    trained_model.eval()
    trained_model.freeze()

    tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

    test_comment = "tai lah semua orang"

    encoding = tokenizer.encode_plus(
        test_comment,
        add_special_tokens = True,
        max_length = 100,
        return_token_type_ids = True,
        padding = "max_length",
        return_attention_mask = True,
        return_tensors = 'pt',
    )


    threshold = 0.5

    test_prediction = trained_model(encoding["input_ids"], 
                                    encoding["token_type_ids"],
                                    encoding["attention_mask"])
    test_prediction = test_prediction.flatten().numpy()

    for label, prediction in zip(labels, test_prediction):
        if prediction < threshold:
            continue
        print(f"{label}: {prediction}")