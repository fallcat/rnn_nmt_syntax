'''
A module implementing various data samplers for datasets.
'''
import numpy as np
import psutil
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


class SequenceLengthSampler3(Sampler):
    ''' A sampler that tries to select batches that have a given total sequence length '''
    def __init__(self, datasource, batch_size, drop_last=False, shuffle=False):
        super(SequenceLengthSampler3, self).__init__(datasource)

        tensor_sentence = [  135,    52,   146,     4,    29,    26,    40,    62,   122,     4,
           13, 27806,     4,    29,    26,    40,   372,   122,   135,   135,
        17800,     4,    29,    26,    40,   167,     4,    29,    26,   146,
          135,   135,   229,    26,    40,    62,   122,     4, 27806,     4,
           29,    26,    40,   372,   122,   135,   135, 17800,     4,    29,
           26,    40,   167,     4,    29,    26,   146,   135,   135,   229,
           26,    40,    62,   122,     4, 27806,     4,    29,    26,    40,
          372,   122,   135,   135, 17800,     4,    29,    26,    40,   167,
            4,    29,    26,   146,   135,   135,   229,    26,    40,    62,
          122,     4, 27806,     4,    29,    26,    40,   372,   122,   135,
          135, 17800,     4,    29,    26,    40,   167,     4,    29,    26,
          146,   135,   135,   229,    26,    40,    62,   122,     4, 27806,
            4,    29,    26,    40,   372,   122,   135,   135, 17800,     4,
          114,   167,   629,    40,   135,   135,    78,  1748,     4,    71,
          995,   281,     4,    29,    26,   167,     4,  6874,    53,   121,
         5950,  5765,   135,   135,  1009,    85,   548,  7881,  7335,     5,
          123,  2274, 10303,  1033,    40,     4, 21369,   135,   135,  3827,
        11626,   110,    91,  5904,  1710,  1607,  6213,   135,   135,  2146,
           36,   807,  7102,     4,    36,   807,  7102,     4,    36,   807,
         7102,   135,   135,    52,   146,     4,    29,    26,    40,    62,
          122,     4, 27806,     4,    29,    26,    40,   372,   122,   135,
          135, 17800,     4,    29,    26,    40,   167,     4,    29,    26,
          146,   135,   135,   229,    26,    40,    62,   122,     4, 27806,
            4,    29,    26,    40,   372,   122,   135,   135, 17800,     4,
          114,   167,   629,    40,   135,   135,  1123,  1225,  1006,    40,
          397,     4,    13,   321,  2917,   578,    35,  9211,   456,   135,
          135,  3854,   122,   145, 13977,   135,   135,  1123,  1225,  1006,
           40,   397,     4,    13,   321,  2917,   578,   197,   135,   135,
           72,  6975,   436,   132,    24,   438,     4,    10,  2901,    97,
          135,   135,    52,   122,   114,    40,   135,   135,   208,     7,
           16,  5698,  4606,   573,   885,   956,   135,   135,     7,    16,
        19701,   956,   135,   135,    48,  2525,  5233, 28904,    98,   135,
          135,   107, 18124,    17, 11267,   328,    17,   620,  3188,  6503,
        10336,   135,   135,    13,   107,    17, 14793,    23,  1010,   197,
        20103,    44,    71,   352,   197,  9945,     4,    29,   299,  6557,
           17,   199,  1208,   135,   135,    43,    26,   308,    27,  1318,
          132,   250,   135,   135,    26,   308,    27,   132,   443,   567,
          135,   135,    26,   308,   132, 26095,     4,  1855,     4,  1658,
            4,  1658,     4,  1658,   135,   135,   626,     4,  1658, 20103,
          174,   629,    45,  1346,  2105,   135,   135,  1123,  1225,  1006,
           40,   397,   135,   135,    13,   321,  2917,   578,    35,  9211,
          456,   135,   135,  3854,   122,   145, 13977,   135,   135,  1123,
         1225,  1006,    40,   397,     4,    13,   321,  2917,   578,    35,
         9211,   456,   135,   135,    72,  6975,   308,   137,   157,   197,
          526,     4,   393,   137,   135,   135,   347,     4,   393,   137,
            4,   393,   137,   135,   135,    52,   382,    40, 13750,    44,
          456,   535,   135,   135,   516, 18273,     4,   183,   131,    40,
        26059,   135,   135,    52,   265,   107, 18124,    17, 11267,   328,
           17,   620,  3188,  6503, 10336,   135,   135,   107,    17, 14793,
           23,  1010,   197, 20103,    44,    71,   352,   197,  9945,     4,
           29,   299,  6557,    17,   199,  1208,   135,   135,    52,   308,
           27,   132,   250,     4,   308,    27,   132,  9746,   135,   135,
           52,   308,   132, 24716,   135,   135,   626,     4,  1658,     4,
         1658,     4,  1658, 20103,   174,   629,    45,  1346,  2105,   135,
          135,    52,   146,     4,    29,    26,    40,    62,   122,     4,
        27806,     4,    29,    26,    40,   372,   122,   135,   135, 17800,
            4,    29,    26,    40,   167,     4,    29,    26,   146,   135,
          135,   229,    26,    40,    62,   122,     4, 27806,     4,    29,
           26,    40,   372,   122,   135,   135, 17800,     4,   114,   167,
          629,    40,   197,    66,  4625,    22,   135,   135,  1123,  1225,
         1006,    40,   397,     4,    13,   321,  2917,   578,    35,  9211,
          456,   135,   135,  3854,   122,   145, 13977,     4,  2126,   135,
          135,  1123,  1225,  1006,    40,   397,     4,    13,   321,  2917,
          578,    35,  9211,   456,   135,   135,   711,    26,   146,     4,
           29,    26,    40,    62,   122,     4, 27806,     4,    29,    26,
           40,   372,   122,   135,   135, 17800,     4,   114,   167,    40,
            4,    29,    26,   146,   135,   135,   229,    26,    40,    62,
          122,     4, 27806,     4,    29,    26,    40,   372,   122,   135,
          135,   229,    26,    40,    62,   122,     4, 27806,     4,    29,
           26,    40,   372,   122,   135,   135, 17800,     4,   114,   167,
           40,     4,    29,    26,   146,   135,   135,   229,    26,    40,
           62,   122,     4, 27806,     4,    29,    26,    40,   372,   122,
          135,   135, 17800,     4,    29,    26,    40,   167,     4,    29,
           26,   146,   135,   135,   229,    26,    40,    62,   122,     4,
        27806,     4,    29,    26,    40,   372,   122,   135,   135, 17800,
            4,   114,   167,    40,     4,    29,    26,   146,   135,   135,
          229,    26,    40,    62,   122,     4, 27806,     4,    29,    26,
           40,   372,   122,   135,   135, 17800,     4,   114,   167,    40,
            4,    29,    26,   146,   135,     2]
        decoded_words = [datasource.index2word[w] for w in tensor_sentence]
        print("decoded words", decoded_words)

        self.batches = []
        self.shuffle = shuffle
        print("len(datasource)", len(datasource))
        print("datasource[0]", datasource[0])

        data_indices = [i[0] for i in sorted(enumerate(datasource), key=lambda x: len(x[1][1]), reverse=True)]
        print("data_indices[0]", data_indices[0])
        print("datasource[data_indices[0]][0]", len(datasource[data_indices[0]][1]))
        print("datasource[data_indices[0]][1]", len(datasource[data_indices[0]][2]))
        # print("example_lengths", datasource[data_indices[0]])
        # print("example_lengths", len(datasource[data_indices[0]][1]))

        i = 0

        batch = []

        for idx in data_indices:
            if len(batch) == 0:
                seq_len = len(datasource[data_indices[i]][1])
                # print("batch_size", batch_size)
                # print("seq_len", seq_len)
                batch_max_len = batch_size // seq_len
                batch_max_len -= batch_max_len % NUM_DEVICES
                # print("batch_max_len", batch_max_len)
            batch.append(idx)
            batch_max_len -= 1
            if batch_max_len <= 0:
                print(seq_len)
                print("batch_max_len", batch_max_len)
                print("batch len", len(batch))
                self.batches.append(batch)
                batch = []

        if not drop_last and len(batch) > 0:
            self.batches.append(batch)
        print("num batches", len(self.batches))
        # print("batch[0]", self.batches[0])

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
            yield self.batches[idx]


class SequenceLengthSampler(Sampler):
    ''' A sampler that tries to select batches that have a given total sequence length '''
    def __init__(self, example_lengths, batch_size, drop_last=False, shuffle=False):
        super(SequenceLengthSampler, self).__init__(example_lengths)

        self.batches = []
        self.shuffle = shuffle

        data_indices = [i[0] for i in sorted(enumerate(example_lengths), key=lambda x: x[1][1], reverse=True)]

        i = 0

        while i < len(data_indices):
            seq_len = example_lengths[data_indices[i]][1]
            batch_max_len = batch_size // seq_len
            batch_max_len -= batch_max_len % NUM_DEVICES
            self.batches.append(data_indices[i:i + batch_max_len])
            i += batch_max_len

        if drop_last and len(self.batches[-1]) < batch_max_len:
            self.batches = self.batches[:-1]
        # print("num batches", len(self.batches))

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


class SequenceLengthSampler2(Sampler):
    ''' A sampler that tries to select batches that have a given total sequence length '''
    def __init__(self, example_lengths, batch_size, drop_last=False, shuffle=False):
        super(SequenceLengthSampler2, self).__init__(example_lengths)

        self.batches = []
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.example_lengths = example_lengths
        self.drop_last = drop_last
        self.indices = [i[0] for i in sorted(enumerate(example_lengths), key=lambda x: x[1][1], reverse=True)]



    def __len__(self):
        ''' Estimate the number of batches per iteration '''
        return len(self.batches)

    def __iter__(self):
        ''' Iterate over the batches '''
        vm = psutil.virtual_memory()
        print("1virtual_memory", vm)
        max_tokens = self.batch_size
        max_sentences = float("Inf")
        bsz_mult = NUM_DEVICES

        batch = []

        def is_batch_full(num_tokens):
            if len(batch) == 0:
                return False
            if len(batch) == max_sentences:
                return True
            if num_tokens > max_tokens:
                return True
            return False

        sample_len = 0
        sample_lens = []
        for idx in self.indices:
            sample_lens.append(self.example_lengths[idx][1])
            sample_len = max(sample_len, sample_lens[-1])
            assert sample_len <= max_tokens, "sentence at index {idx} exceeds max_tokens limit!".format(idx=idx)
            num_tokens = (len(batch) + 1) * sample_len
            if is_batch_full(num_tokens):
                mod_len = max(
                    bsz_mult * (len(batch) // bsz_mult),
                    len(batch) % bsz_mult,
                )
                vm = psutil.virtual_memory()
                print("2virtual_memory", vm)
                yield batch[:mod_len]
                batch = batch[mod_len:]
                sample_lens = sample_lens[mod_len:]
                sample_len = max(sample_lens) if len(sample_lens) > 0 else 0

            batch.append(idx)

        if len(batch) > 0 and not self.drop_last:
            yield batch