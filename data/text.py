import torch
import collections
import itertools
from torch import nn
from torch.utils.data import Dataset

import tarfile
import torch
import torch.utils.data as Data
from model import EOS_token, DEVICE, UNK_token

PAD = '<PAD>'
# ALIGN = '<ALIGN>'
SOS = '<SOS>'
EOS = '<EOS>'
UNK = '<UNK>'


class TextDataset(Dataset):
    """
    Prepare data from WMTDataset
    """
    def __init__(self, max_length, span_size, sort, filter, split="train", reverse=False):
        self.word2index = {PAD: 0, SOS: 1, EOS: 2, UNK: 3}
        self.word2count = {}
        self.index2word = {0: PAD, 1: SOS, 2: EOS, 3: UNK}
        self.num_words = 4  # Count SOS and EOS and UNK
        self.split = split
        self.sort = sort
        self.filter = filter

        self.span_size = span_size
        self.reverse = reverse
        self.max_length = max_length

        self.pairs = []
        self.prepare_data()

    def __len__(self):
        ''' Get the length of the dataset '''
        return len(self.pairs)

    def __getitem__(self, index):
        ''' Get the story/stories at the specified index/indices '''
        if isinstance(index, collections.Sequence):
            return tuple(
                tuple([i]) + tuple(self.tensors_from_pair(self.pairs[i])) for i in index
                # tuple([i]) + tuple(torch.LongTensor(s) for s in self.pairs[i]) for i in index
            )
        else:
            return tuple([index]) + tuple(self.tensors_from_pair(self.pairs[index]))

    @property
    def padding_idx(self):
        ''' Return the padding value '''
        return self.word2index[PAD]

    @property
    def sos_idx(self):
        ''' Return the start of summary value '''
        return self.word2index[SOS]

    @property
    def eos_idx(self):
        ''' Return the end of summary value '''
        return self.word2index[EOS]

    @property
    def unk_idx(self):
        ''' Return the end of summary value '''
        return self.word2index[UNK]

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def prepare_data(self):
        self.read_langs()
        print("Counting words from vocab file...")
        self.read_vocab()
        print("Counted words:", self.num_words)

    def read_vocab(self):
        ''' Read in the vocabulary file '''
        raise NotImplementedError('Subclasses must implement preprocess!')

    def read_langs(self):
        ''' Read the texts of two languages '''
        raise NotImplementedError('Subclasses must implement preprocess!')

    def filter_pair(self, p):
        return len(p[0].split(' ')) < self.max_length - (self.span_size + 1) and \
               len(p[1].split(' ')) < self.max_length - (self.span_size + 1)

    def filter_pairs(self, pairs):
        return [pair for pair in pairs if self.filter_pair(pair)]

    def indexes_from_sentence(self, sentence):
        return [self.word2index[word] if word in self.word2index else UNK_token
                for word in sentence.split(' ')]

    def tensor_from_sentence(self, sentence):
        indexes = self.indexes_from_sentence(sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long) #.view(-1, 1)

    def tensors_from_pair(self, pair):
        input_tensor = self.tensor_from_sentence(pair[0])
        target_tensor = self.tensor_from_sentence(pair[1])
        return input_tensor, target_tensor

    def collate(self, data, sort=False):
        ''' Collate the data into a batch '''
        if not data:
            return []

        def make_batch(example_ids, inputs, targets):
            ''' Make a batch given a list of inputs and targets '''
            # must store off lengths before padding sequence
            input_lens = torch.LongTensor([len(input) for input in inputs])
            target_lens = torch.LongTensor([len(target) for target in targets])

            batch_size = len(target_lens)
            span_seq_len = int(
                (len(max(targets, key=lambda x: len(x))) - 1) / self.span_size) + 1

            dummy_data = torch.ones((span_seq_len * self.span_size), dtype=torch.long)

            inputs = nn.utils.rnn.pad_sequence(
                inputs, batch_first=True, padding_value=self.padding_idx)
            targets = nn.utils.rnn.pad_sequence(
                [dummy_data] + list(targets), batch_first=True, padding_value=self.padding_idx)[1:]

            return {
                'inputs': inputs,
                'input_lens': input_lens,
                'targets': targets,
                'target_lens': target_lens,
                'example_ids': example_ids,
                'batch_size': batch_size,
                'span_seq_len': span_seq_len
            }

        if any(
                isinstance(d, tuple) and len(d) and
                isinstance(d[0], collections.Sequence)
                for d in data
        ):
            if sort:
                # Sort within each chunk
                data = [sorted(d, key=lambda x: len(x[1]), reverse=True) for d in data]

            batch = make_batch(*zip(*list(itertools.chain.from_iterable(data))))
            batch['chunk_sizes'] = [len(l) for l in data]
            return batch
        else:
            if sort:
                data = sorted(data, key=lambda x: len(x[1]), reverse=True)

            return make_batch(*zip(*data))
