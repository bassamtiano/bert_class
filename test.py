import torch
# from cleantext import clean
from transformers import BertTokenizer
from sklearn.metrics import classification_report

# #provide string with emojis
# text = "Mpok sylvi udah mulai emosi\ud83d\ude02\n #DebatFinalPilkadaJKT"

# #print text after removing the emojis from it
# text = clean(text, no_emoji=True)

# tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')
# hasil = tokenizer(text)
# print(hasil)

# y_true = [[0, 1, 0, 0]]
# y_pred = [[0, 1, 0, 0]]

# target_names = []
# report = classification_report(y_true, 
#                                y_pred)
# print(report)

data = torch.rand(100, 2)

print(data.shape)

train, test = torch.utils.data.random_split(data, [data.shape[0] * 0.8, data.shape[0] * 0.2], generator = torch.Generator().manual_seed(42))
print(train.shape)
print(test.shape)
