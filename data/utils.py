from functools import partial
from data.sampler2 import RandomBatchSampler, SequenceLengthSampler

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler


def get_dataloader(dataset, config, split, worker_init_fn=None, pin_memory=True, num_devices=1, shuffle=False):
    ''' Utility function that gets a data loader '''
    dataset = dataset(config['max_length'], config['span_size'], config['filter'], split, config['trim'])
    # if config['batch_method'] == 'token':
    #     # Calculate batch sizes for each device. Potentially reduce the batch size on device 0 as
    #     # the optimization step (all the gradients from all devices) happens on device 0.
    #     batch_sizes = [config['minibatch_size'] - config['batch_size_buffer']]
    #     batch_sizes += [config['minibatch_size']] * (num_devices - 1)
    #     batch_sampler = SequenceLengthSampler(
    #         batch_sizes,
    #         [tuple(len(p) for p in s) for s in dataset.pairs],
    #         shuffle=shuffle
    #     )

    if config['batch_method'] == 'token':
        batch_sampler = SequenceLengthSampler(
            dataset,
            config['minibatch_size'],
            config['drop_last'],
            config['shuffle']
        )
    elif config['batch_method'] == 'random_batch':
        batch_sampler = RandomBatchSampler(
            dataset,
            config['minibatch_size'],
            config['drop_last'],
            config['shuffle']
        )
    elif config['batch_method'] == 'example':
        sampler_fn = RandomSampler if shuffle else SequentialSampler
        # print("shuffle", shuffle)
        batch_sampler = BatchSampler(
            sampler_fn(dataset),
            config['minibatch_size'],
            config['drop_last']
    )
    else:
        raise ValueError('Unknown batch method!')

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=partial(dataset.collate, sort=True),
        num_workers=1,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn
    )