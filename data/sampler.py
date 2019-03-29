'''
A module implementing various data samplers for datasets.
'''
import numpy as np
from torch.utils.data import Sampler


class SequenceLengthSampler(Sampler):
    ''' A sampler that tries to select batches that have a given total sequence length '''
    def __init__(self, max_lengths, example_lengths, shuffle=False):
        '''
        Initializer the sequence length sampler

        Inputs:
        max_lengths - a list of lengths of the desired total sequence length for each device
        lengths - a list containing the length for each example in the dataset
        '''
        super(SequenceLengthSampler, self).__init__(example_lengths)

        self.batches = []
        self.shuffle = shuffle

        def pairwise_max(x, y):
            ''' Compute the pair-wise maximums '''
            return tuple(max(l) for l in zip(x, y))

        def batch_max(batch):
            ''' Compute the pair-wise maximum lengths over the batch '''
            max_lengths = (0, 0)
            for _, lengths in batch:
                max_lengths = pairwise_max(lengths, max_lengths)

            return max_lengths

        def split(batch):
            ''' Split a batch across devices '''
            device_batches = []
            example_length = sum(batch_max(batch))
            for max_device_length in max_lengths:
                max_idx = max_device_length // example_length
                ids, _ = zip(*batch[:max_idx])
                device_batches.append(ids)
                batch = batch[max_idx:]

                if not batch:
                    break

            return device_batches, batch

        next_batch = []
        max_batch_lengths = (0, 0)
        max_batch_length = sum(max_lengths)
        for idx, lengths in sorted(enumerate(example_lengths), key=lambda x: x[1]):
            expected_batch_lengths = pairwise_max(lengths, max_batch_lengths)
            expected_batch_length = sum(expected_batch_lengths) * (len(next_batch) + 1)
            if expected_batch_length > max_batch_length:
                device_batches, next_batch = split(next_batch)
                self.batches.append(device_batches)
                max_batch_lengths = batch_max(next_batch)

            max_batch_lengths = pairwise_max(lengths, max_batch_lengths)
            next_batch.append((idx, lengths))

        # There is always at least one left over batch since it's the last step in the loop, so make
        # sure to add it to the list of batches.
        def steal(batch):
            # First see if you can steal from the passed in batch.
            if batch:
                return batch.pop()

            # Then try to steal from the end of the list of batches. They have the longest examples.
            for batch_group in reversed(self.batches):
                for batch_idx, batch in enumerate(batch_group):
                    if len(batch) > 1:
                        idx = batch[-1]
                        batch_group[batch_idx] = batch[:-1]
                        return idx, example_lengths[idx]

            raise RuntimeError('Cannot fill batch... degenerate dataset.')

        num_devices = len(max_lengths)
        while next_batch:
            device_batches, next_batch = split(next_batch)

            # Make sure there are enough elements to split amongst the available devices.
            while len(device_batches) < num_devices:
                idx, lengths = steal(next_batch)
                device_batches.append([idx])

            self.batches.append(device_batches)

    def __len__(self):
        ''' Estimate the number of batches per iteration '''
        return len(self.batches)

    def __iter__(self):
        ''' Iterate over the batches '''
        indices = np.arange(len(self))
        if self.shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            yield self.batches[idx]
