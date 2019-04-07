'''
A module implementing various data samplers for datasets.
'''
import numpy as np
from model import NUM_DEVICES
from torch.utils.data import Sampler


class RandomBatchSampler(Sampler):
    ''' A sampler that tries to select batches that have a given total sequence length '''
    def __init__(self, datasource, batch_size, drop_last=False, shuffle=False):
        super(RandomBatchSampler, self).__init__(datasource)

        self.batches = []
        self.shuffle = shuffle

        data_indices = [i[0] for i in sorted(enumerate(datasource), key=lambda x: len(x[1]), reverse=True)]

        batch = []

        for idx in data_indices:
            batch.append(idx)
            if len(batch) == batch_size:
                self.batches.append(batch)
                batch = []

        if not drop_last and len(batch) > 0:
            self.batches.append(batch)

    def __len__(self):
        ''' Estimate the number of batches per iteration '''
        return len(self.batches)

    def __iter__(self):
        ''' Iterate over the batches '''
        batch_indices = np.arange(len(self))
        if self.shuffle:
            np.random.shuffle(batch_indices)

        for idx in batch_indices:
            yield self.batches[idx]


class SequenceLengthSampler(Sampler):
    ''' A sampler that tries to select batches that have a given total sequence length '''
    def __init__(self, example_lengths, batch_size, drop_last=False, shuffle=False):
        super(SequenceLengthSampler, self).__init__(example_lengths)

        self.batches = []
        self.shuffle = shuffle

        data_indices = [i[0] for i in sorted(enumerate(example_lengths), key=lambda x: x[1][1], reverse=True)]
        # print("total indices", len(data_indices), len(example_lengths))
        # print("data_indices", [example_lengths[i] for i in data_indices[:100]])


        i = 0

        while i < len(data_indices):
            # print("i", i)
            print(len(self.batches))
            seq_len = example_lengths[data_indices[i]][1]
            # print("batch_size", batch_size)
            # print("seq_len", seq_len)
            # print(example_lengths[data_indices[i]])
            batch_max_len = batch_size // seq_len
            # print("batch_max_len", batch_max_len)
            # print("NUM_DEVICES", NUM_DEVICES)
            batch_max_len -= batch_max_len % NUM_DEVICES
            # print("batch_max_len", batch_max_len)
            # print("batch_max_len", batch_max_len)
            self.batches.append(data_indices[i:i + batch_max_len])
            i += batch_max_len

        if drop_last and len(self.batches[-1]) < batch_max_len:
            self.batches = self.batches[:-1]
        print("num batches", len(self.batches))

    def __len__(self):
        ''' Estimate the number of batches per iteration '''
        return len(self.batches)

    def __iter__(self):
        ''' Iterate over the batches '''
        batch_indices = np.arange(len(self))
        if self.shuffle:
            np.random.shuffle(batch_indices)

        for idx in batch_indices:
            # print("idx", idx)
            # print(len(self.batches[idx]))
            yield self.batches[idx]