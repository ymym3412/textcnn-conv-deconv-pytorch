from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm

from collections import Counter
from copy import deepcopy


def load_hotel_review_data(path, sentence_len):
    """
    Load Hotel Reviews data from pickle distributed in https://drive.google.com/file/d/0B52eYWrYWqIpQzhBNkVxaV9mMjQ/view
    This file is published in https://github.com/dreasysnail/textCNN_public
    
    :param path: pickle path
    :return: 
    """
    import _pickle as cPickle
    with open(path, "rb") as f:
        data = cPickle.load(f, encoding="latin1")

    train_data, test_data = HotelReviewsDataset(data[0], deepcopy(data[2]), deepcopy(data[3]), sentence_len, transform=ToTensor()), \
                             HotelReviewsDataset(data[1], deepcopy(data[2]), deepcopy(data[3]), sentence_len, transform=ToTensor())
    return train_data, test_data


class HotelReviewsDataset(Dataset):
    """
    Hotel Reviews Dataset
    """
    def __init__(self, data_list, word2index, index2word, sentence_len, transform=None):
        self.word2index = word2index
        self.index2word = index2word
        self.n_words = len(self.word2index)
        self.data = data_list
        self.sentence_len = sentence_len
        self.transform = transform
        self.word2index["<PAD>"] = self.n_words
        self.index2word[self.n_words] = "<PAD>"
        self.n_words += 1
        print(self.index2word)
        temp_list = []
        for sentence in tqdm(self.data):
            if len(sentence) > self.sentence_len:
                # truncate sentence if sentence length is longer than `sentence_len`
                temp_list.append(np.array(sentence[:self.sentence_len]))
            else:
                # pad sentence  with '<PAD>' token if sentence length is shorter than `sentence_len`
                sent_array = np.lib.pad(np.array(sentence),
                                        (0, self.sentence_len - len(sentence)),
                                        "constant",
                                        constant_values=(self.n_words-1, self.n_words-1))
                temp_list.append(sent_array)
        self.data = np.array(temp_list, dtype=np.int32)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform:
            data = self.transform(data)
        return data

    def vocab_lennght(self):
        return len(self.word2index)


class TextClassificationDataset(Dataset):
    def __init__(self, data_path, label_path, tokenized, sentence_len=60, transoform=None):
        self.word2index = {"<PAD>": 0, "<UNK>": 1}
        self.index2word = {0: "<PAD>", 1: "<UNK>"}
        self.n_words = 2
        self.sentence_len = sentence_len
        # Data load
        with open(data_path, encoding="utf-8") as f:
            data = [line.split() for line in f]

        if tokenized == "mecab":
            # replace low frequency word to UNK token
            word_bucket = []
            for sentence in data:
                word_bucket.extend(sentence)
            cnt = Counter(word_bucket)
            rare_word = []
            for common in cnt.most_common():
                if common[1] <= 2:
                    rare_word.append(common[0])
            print("Rare word")
            rare_word = set(rare_word)
            print(len(rare_word))

            for sentence in data:
                for word in sentence:
                    if word in rare_word:
                        continue
                    elif word not in self.word2index:
                        self.word2index[word] = self.n_words
                        self.index2word[self.n_words] = word
                        self.n_words += 1
            # Transform to idx
            self.data = np.array([[self.word2index[word]
                                   if word not in rare_word
                                   else self.word2index["<UNK>"] for word in sentence]
                                  for sentence in tqdm(data)])
        elif tokenized == "sentencepiece":
            for sentence in data:
                # remove meta symbol
                # TODO:this process remove blank which in sentene. Are there other method?
                for word in map(lambda word: word.replace("▁", ""), sentence):
                    if word not in self.word2index:
                        self.word2index[word] = self.n_words
                        self.index2word[self.n_words] = word
                        self.n_words += 1
            self.data = np.array([[self.word2index[word] for word in map(lambda word: word.replace("▁", ""), sentence)]
                                  for sentence in tqdm(data)])

        temp_list = []
        for sentence in self.data:
            if len(sentence) > self.sentence_len:
                # truncate sentence if sentence length is longer than `sentence_len`
                temp_list.append(np.array(sentence[:self.sentence_len]))
            else:
                # pad sentence  with '<PAD>' token if sentence length is shorter than `sentence_len`
                sent_array = np.lib.pad(np.array(sentence),
                                        (0, self.sentence_len - len(sentence)),
                                        "constant",
                                        constant_values=(0, 0))
                temp_list.append(sent_array)
        self.data = np.array(temp_list, dtype=np.int32)
        with open(label_path, encoding="utf-8") as f:
            self.labels = np.array([np.array([int(label)]) for label in f], dtype=np.int32)
        self.transform = transoform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        label = self.labels[idx]
        sample = {"sentence": sentence, "label": label}

        if self.transform:
            sample = {"sentence": self.transform(sample["sentence"]),
                      "label": self.transform(sample["label"])}

        return sample

    def vocab_length(self):
        return self.n_words


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, data):
        return torch.from_numpy(data).type(torch.LongTensor)
