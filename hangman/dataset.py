import json
import torch
import math
import numpy as np

class DictionaryDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = self.load_data("./hangman/words_250000_train.txt")

    def load_data(self, dictionary_file_location):
        text_file = open(dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word = self.data[idx]
        obscured_word = '_'*len(word)
        word_feature_tensor = encode_word(word)
        obscured_tensor = encode_word(obscured_word)
        word_arr = np.zeros(26, dtype=np.float32)
        for i in word: # np.put(word_arr, [ord(i)-97 for i in word], [1]*len(word))
            word_arr[ord(i)-97] = 1
        return obscured_tensor,word_feature_tensor,word_arr

    def collate_fn(self,batch):
        obscured_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(row[0],dtype=torch.float32) for row in batch],batch_first=True,padding_value=26)
        word_feature_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(row[1],dtype=torch.float32) for row in batch],batch_first=True, padding_value=26)
        word_tensor = torch.stack([torch.tensor(row[2],dtype=torch.float32) for row in batch]) #TODO: list to tensor slow process improve
        return obscured_tensor,word_feature_tensor,word_tensor


def encode_word(word):
    full_word = [26 if i=='_' else ord(i)-97 for i in word]
    word_tensor = np.zeros((len(word), 27), dtype=np.float32)
    for i,j in enumerate(full_word):
        word_tensor[i,j] = 1
    return word_tensor

def load_dataset():
    dataset = DictionaryDataset()
    n_dataset = len(dataset)
    n_dev = math.ceil(len(dataset)*0.15)
    n_test = math.ceil(len(dataset)*0.15)
    n_train = n_dataset - n_test - n_dev
    train_dataset, dev_dataset, test_dataset = torch.utils.data.random_split(dataset, (n_train, n_dev, n_test))
    collate_fn = dataset.collate_fn
    return train_dataset,dev_dataset,test_dataset,collate_fn
