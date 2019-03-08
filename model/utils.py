import os
import time
import math
import torch
import argparse
import shutil

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
                            help='Record this run in experiment')

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
