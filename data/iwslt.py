import tarfile
import torch
import torch.utils.data as Data
# from model import EOS_token, DEVICE, UNK_token
from rnn_nmt_syntax.model import EOS_token, DEVICE, UNK_token



class IWSLTDataset(object):
    """
    Prepare data from WMTDataset
    """
    def __init__(self, max_length, reverse=False):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.num_words = 3  # Count SOS and EOS and UNK
        self.dir_path = "/content/gdrive/My Drive/data/"
        self.vocab_file = 'vocab'
        self.splits = {
            'valid': 'dev.tok',
            'train': 'train.tok'
        }
        self.reverse = reverse
        self.max_length = max_length

        self.pairs = {}
        self.prepare_data()

    def read_vocab(self):
        vocab = open(self.dir_path+self.vocab_file, 'r').read().strip().split('\n')
        for v in vocab:
            self.add_word(v)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def read_langs(self):
        print("Reading lines...")

        t = tarfile.open(self.tar_path, "r")

        for split in self.splits:
            en_lines = open(self.dir_path + '%s.en' % (self.splits[split])).read().strip().split('\n')
            de_lines = open(self.dir_path + '%s.de' % (self.splits[split])).read().strip().split('\n')

            # Split every line into pairs
            pairs = [[s1, s2] for s1, s2 in zip(de_lines, en_lines)]

            # Reverse pairs, make Lang instances
            if self.reverse:
                pairs = [list(reversed(p)) for p in pairs]

            print("Read %s sentence pairs in %s" % (len(pairs), split))

            pairs = self.filter_pairs(pairs)
            print("Trimmed to %s sentence pairs" % len(pairs))

            self.pairs[split] = pairs

    def prepare_data(self):
        self.read_langs()
        print("Counting words from vocab file...")
        self.read_vocab()
        print("Counted words:", self.num_words)

    def filter_pair(self, p):
        return len(p[0].split(' ')) < self.max_length and \
               len(p[1].split(' ')) < self.max_length

    def filter_pairs(self, pairs):
        return [pair for pair in pairs if self.filter_pair(pair)]

    def indexes_from_sentence(self, sentence):
        return [self.word2index[word] if word in self.word2index else self.word2index[UNK_token]
                for word in sentence.split(' ')]

    def tensor_from_sentence(self, sentence):
        indexes = self.indexes_from_sentence(sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)

    def tensors_from_pair(self, pair):
        input_tensor = self.tensor_from_sentence(pair[0])
        target_tensor = self.tensor_from_sentence(pair[1])
        return input_tensor, target_tensor
