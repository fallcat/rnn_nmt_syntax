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
import torch.nn as nn
import torch.nn.functional as F
from sacremoses import MosesDetokenizer
# from torch.optim.lr_scheduler import _LRScheduler

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker


class LabelSmoothingLoss(nn.Module):
    '''
    Implements the label smoothing loss as defined in
    https://arxiv.org/abs/1512.00567

    The API for this loss is modeled after nn..CrossEntropyLoss:

    1) The inputs and targets are expected to be (B x C x ...), where B is the batch dimension, and
    C is the number of classes
    2) You can pass in an index to ignore
    '''
    def __init__(self, smoothing=0.0, ignore_index=-1, reduction='sum'):
        ''' Initialize the label smoothing loss '''
        super(LabelSmoothingLoss, self).__init__()

        self.reduction = reduction
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, inputs, targets): # pylint:disable=arguments-differ
        ''' The implements the actual label smoothing loss '''
        num_classes = inputs.shape[1]
        smoothed = inputs.new_full(inputs.shape, self.smoothing / num_classes)
        smoothed.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        if self.ignore_index >= 0 and self.ignore_index < num_classes:
            smoothed[:, self.ignore_index] = 0.

            mask = targets == self.ignore_index
            smoothed.masked_fill_(mask.unsqueeze(1), 0.)

        return F.kl_div(inputs.log_softmax(1), smoothed, reduction=self.reduction)


class Parallel(nn.Sequential):
    '''
    A container similar to torch.nn.Sequential, but returns a tuple of outputs from the modules
    added to the container, rather than return the output of sequentially applying the modules.
    '''
    def forward(self, *args, **kwargs): # pylint:disable=arguments-differ
        outputs = []
        for module in self._modules.values():
            outputs.append(module(*args, **kwargs))
        return outputs


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


def save_predictions(preds, evaluate_path, detokenize):
    md = MosesDetokenizer()
    with open(evaluate_path, 'w') as f:
        for pred in preds:
            if '<EOS>' in pred:
                pred = pred[:pred.index('<EOS>')]
            if detokenize:
                # print("pred", pred)
                output = md.detokenize(' '.join(pred).replace('@@ ', '').split())
            else:
                output = ' '.join(pred)
            f.write(output + '\n')

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
    print("inputs", len(inputs))
    for inp in inputs:
        print("type:", type(inp))
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