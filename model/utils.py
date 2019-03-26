import io
import os
import time
import math
import tqdm
import torch
import argparse
import contextlib
import sys
import shutil
import random
import collections, gc, torch
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def save_checkpoint(state, is_best, filename='experiments/exp01/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'experiments/exp01/model_best.pth.tar')


def restore_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, restore):
    if restore is not None:
        if os.path.isfile(restore):
            print("=> loading checkpoint '{}'".format(restore))
            checkpoint = torch.load(restore)
            start_iter = checkpoint['epoch']
            encoder.load_state_dict(checkpoint['encoder_state'])
            decoder.load_state_dict(checkpoint['decoder_state'])
            encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
            print("=> loaded checkpoint '{}' (iter {})".format(restore, checkpoint['epoch']))
            checkpoint_loaded = True
        else:
            print("=> no checkpoint found at '{}'".format(restore))


def get_cl_args():
    """Get the command line arguments using argparse."""
    arg_parser = argparse.ArgumentParser(description='Train machine translation model with RNN + Syntax')

    arg_parser.add_argument('-s', '--save', action='store', default='experiments/exp02/checkpoint.pth.tar',
                            help='Specify the path of checkpoint to save the stored model')

    arg_parser.add_argument('-b', '--best-model', action='store', default='experiments/exp02/model_best.pth.tar',
                            help='Specify the path of checkpoint to save the best stored model')

    arg_parser.add_argument('-p', '--plot', action='store', default='experiments/exp02/plot.pdf',
                            help='Specify the path to save to plot')

    arg_parser.add_argument('-r', '--restore', action='store', default=None,
                            help='Specify the path of checkpoint to load the stored model')

    arg_parser.add_argument('--hidden-size', action='store', type=int, default=1000,
                            help='Specify the hidden size of the model')

    arg_parser.add_argument('-d', '--dropout', action='store', type=float, default=0.2,
                            help='Specify the dropout rate of the model')

    arg_parser.add_argument('-l', '--num-layers', action='store', type=int, default=4,
                            help='Specify the number of GRU layers of the model')

    arg_parser.add_argument('-e', '--num-epochs', action='store', type=int, default=2,
                            help='Specify the number of epochs to train')

    arg_parser.add_argument('-i', '--num-iters', action='store', type=int, default=50000,
                            help='Specify the number of iterations each epoch to train')

    arg_parser.add_argument('--max-length', action='store', type=int, default=51,
                            help='Specify the max length of a sentence')

    arg_parser.add_argument('--span-size', action='store', type=int, default=3,
                            help='Specify the span size of the model')

    arg_parser.add_argument('--learning-rate', action='store', type=float, default=0.01,
                            help='Specify the learning rate')

    arg_parser.add_argument('--weight-decay', action='store', type=float, default=1e-5,
                            help='Specify the weight decay')

    arg_parser.add_argument('--print-every', action='store', type=int, default=40,
                            help='Specify the number of batches to report loss')

    arg_parser.add_argument('--plot-every', action='store', type=int, default=1,
                            help='Specify the number of iterations to record loss and print later')

    arg_parser.add_argument('--teacher-forcing-ratio', action='store', type=float, default=0.5,
                            help='Specify the percent of training to do teacher forcing'
                                 ' - use ground truth target words instead of generated'
                                 'words to predict next words')

    arg_parser.add_argument('--train-size', action='store', type=int, default=None,
                            help='Specify the size of data to train. If specify a small number,'
                                 'can try to make the model converge before training on larger data.')

    arg_parser.add_argument('--minibatch-size', action='store', type=int, default=128,
                            help='Specify the size of minibatch')

    arg_parser.add_argument('--do-experiment', action='store_true',
                            help='Record this run in experiment')

    arg_parser.add_argument('--num-evaluate', action='store', type=int, default=10,
                            help='Number of sentences to evaluate during training')

    arg_parser.add_argument('--evaluate-every', action='store', type=int, default=10,
                            help='Evaluate every x epochs')

    arg_parser.add_argument('--optimizer', action='store', type=str, default="SGD",
                            help='Specify which optimizer to use')

    arg_parser.add_argument('--dataset', action='store', type=str, default="WMT",
                            help='Specify which data to use')

    arg_parser.add_argument('--mode', action='store', type=str, default="train",
                            help='Specify train or evaluate, if evaluate, need to load a model')

    arg_parser.add_argument('--evaluate-path', action='store', type=str, default="experiments/exptest/translated.txt",
                            help='Specify a path to store the evaluated sentences')

    arg_parser.add_argument('--seed', action='store', type=int, default=None,
                            help='Set seed for random scheduler')

    arg_parser.add_argument('--shuffle', action='store', type=bool, default=True,
                            help='Shuffle the dataloader')

    return arg_parser.parse_args()


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def save_plot(points, plot_path):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(plot_path)


def debug_memory():
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in tensors.items():
        print('{}\t{}'.format(*line))
    print("--------")


def save_predictions(preds, evaluate_path):
    with open(evaluate_path, 'w') as f:
        for pred in preds:
            if 'EOS' in pred:
                f.write(' '.join(pred[:pred.index('EOS')]) + '\n')
            else:
                f.write(' '.join(pred) + '\n')

# Beam search utils

# Recursively split or chunk the given data structure. split_or_chunk is based on
# torch.nn.parallel.scatter_gather.gather
def cat(outputs, dim=0):
    r"""
    Concatenates tensors recursively in collections.
    """
    def cat_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return torch.cat(outputs, dim=dim)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, cat_map([d[k] for d in outputs]))
                              for k in out))
        return type(out)(map(cat_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return cat_map(outputs)
    finally:
        cat_map = None


# Recursively split or chunk the given data structure. split_or_chunk is based on
# torch.nn.parallel.scatter_gather.scatter
def split_or_chunk(inputs, num_chunks_or_sections, dim=0):
    r"""
    Splits tensors into approximately equal chunks or specified chunk sizes (based on the
    'num_chunks_or_sections'). Duplicates references to objects that are not tensors.
    """
    def split_map(obj):
        if isinstance(obj, torch.Tensor):
            if isinstance(num_chunks_or_sections, int):
                return torch.chunk(obj, num_chunks_or_sections, dim=dim)
            else:
                return torch.split(obj, num_chunks_or_sections, dim=dim)
        if isinstance(obj, tuple) and obj:
            return list(zip(*map(split_map, obj)))
        if isinstance(obj, list) and obj:
            return list(map(list, zip(*map(split_map, obj))))
        if isinstance(obj, dict) and obj:
            return list(map(type(obj), zip(*map(split_map, obj.items()))))
        if isinstance(num_chunks_or_sections, int):
            return [obj for chunk in range(num_chunks_or_sections)]
        else:
            return [obj for chunk in num_chunks_or_sections]

    # After split_map is called, a split_map cell will exist. This cell
    # has a reference to the actual function split_map, which has references
    # to a closure that has a reference to the split_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return split_map(inputs)
    finally:
        split_map = None

# beam search utils end


def get_random_seed_fn(seed, cuda=True):
    ''' Return a function that sets a random seed '''
    def set_random_seed(worker_id=0): # pylint:disable=unused-argument
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    return set_random_seed


@contextlib.contextmanager
def tqdm_wrap_stdout():
    ''' Wrap a sys.stdout and funnel it to tqdm.write '''
    saved = sys.stdout
    sys.stdout = TQDMStreamWrapper(sys.stdout)
    yield
    sys.stdout = saved


class TQDMStreamWrapper(io.IOBase):
    ''' A wrapper around an existing IO stream to funnel to tqdm '''
    def __init__(self, stream):
        ''' Initialize the stream wrapper '''
        super(TQDMStreamWrapper, self).__init__()
        self.stream = stream

    def write(self, line):
        ''' Potentially write to the stream '''
        if line.rstrip(): # avoid printing empty lines (only whitespace)
            tqdm.write(line, file=self.stream)