import os
import torch
import ipdb

class Dictionary(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = []

    def add_char(self, char):
        if char not in self.char2idx:
            self.idx2char.append(char)
            self.char2idx[char] = len(self.idx2char) - 1
        return self.char2idx[char]

    def __len__(self):
        return len(self.idx2char)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        assert os.path.exists(path)
        # Create the dictionary and count the number of tokens (characters)

        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                #print line
                #words = line.split() + ['<eos>']
                chars = list(line)+['<eos>']
                tokens += len(chars)
                #print char
                for char in chars:
                    self.dictionary.add_char(char)

        # initialize the ids LongTensor and put id of words
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                chars = list(line) + ['<eos>']
                for char in chars:
                    ids[token] = self.dictionary.char2idx[char]
                    token += 1
        return ids

#path = "data/penn"
#ipdb.set_trace()
#corpus = Corpus(path)


