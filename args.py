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

    arg_parser.add_argument('--lr-decay', action='store', type=float, default=1,
                            help='Multiplicative factor of learning rate decay.')

    arg_parser.add_argument(
        '--profile-cuda-memory',
        default=False,
        const='cuda.prof',
        nargs='?',
        type=str,
        help='Whether to profile CUDA memory.'
    )

    arg_parser.add_argument(
        '--batch-size-buffer',
        type=int,
        default=0,
        help='By how many tokens to reduce the batch size on the GPU of the optimizer'
    )

    arg_parser.add_argument(
        '--batch-method',
        type=str,
        default='token',
        choices=['token', 'example'],
        help='By which method to sample batches'
    )

    return arg_parser.parse_args()
