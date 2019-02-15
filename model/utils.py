import os
import time
import math
import torch
import argparse
import shutil


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
    '''Get the command line arguments using argparse.'''
    arg_parser = argparse.ArgumentParser(description='Train machine translation model with RNN + Syntax')

    arg_parser.add_argument('-s', '--save', action='store', default='experiments/exp02/checkpoint.pth.tar',
                            help='Specify the path of checkpoint to save the stored model')

    arg_parser.add_argument('-r', '--restore', action='store', default=None,
                            help='Specify the path of checkpoint to load the stored model')

    arg_parser.add_argument('--hidden-size', action='store', type=int, default=1000,
                            help='Specify the hidden size of the model')

    arg_parser.add_argument('-d', '--dropout', action='store', type=float, default=0.2,
                            help='Specify the dropout rate of the model')

    arg_parser.add_argument('-l', '--num-layers', action='store', type=int, default=4,
                            help='Specify the number of GRU layers of the model')

    return arg_parser.parse_args()
