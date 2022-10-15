import pandas as pd
from torch.utils.data import DataLoader, Dataset


if __name__ == '__main__':
    data_s = pd.read_csv('data/preprocessed_indonesian_toxic_tweet.csv')

    print(data_s.keys()[1:])