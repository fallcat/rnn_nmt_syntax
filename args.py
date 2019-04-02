import argparse


def add_lstm_args(parser):
    group = parser.add_argument_group('LSTM Model')

    group.add_argument('--hidden-size', action='store', type=int, default=1000,
                            help='Specify the hidden size of the model')

    group.add_argument('-d', '--dropout', action='store', type=float, default=0.2,
                            help='Specify the dropout rate of the model')

    group.add_argument('-l', '--num-layers', action='store', type=int, default=4,
                            help='Specify the number of GRU layers of the model')

    return group


def add_train_args(parser):
    group = parser.add_argument_group('Training')

    group.add_argument('-e', '--num-epochs', action='store', type=int, default=2,
                            help='Specify the number of epochs to train')

    group.add_argument('--learning-rate', action='store', type=float, default=0.01,
                            help='Specify the learning rate')

    group.add_argument('--weight-decay', action='store', type=float, default=1e-5,
                            help='Specify the weight decay')

    group.add_argument('--print-every', action='store', type=int, default=40,
                            help='Specify the number of batches to report loss')

    group.add_argument('--num-evaluate', action='store', type=int, default=10,
                            help='Number of sentences to evaluate during training')

    group.add_argument('--train-size', action='store', type=int, default=None,
                       help='Specify the size of data to train. If specify a small number,'
                            'can try to make the model converge before training on larger data.')

    group.add_argument('--minibatch-size', action='store', type=int, default=128,
                       help='Specify the size of minibatch')

    group.add_argument('--optimizer', action='store', type=str, default="SGD",
                            choices=["SGD", "Adadelta", "Adagrad", "RMSprop", "Adam"],
                            help='Specify which optimizer to use')

    group.add_argument('--lr-decay', action='store', type=float, default=1,
                            help='Multiplicative factor of learning rate decay.')

    return group


def add_evaluate_args(parser):
    group = parser.add_argument_group('Evaluate')

    group.add_argument('--evaluate-path', action='store', type=str, default="translated.txt",
                            help='Specify a path to store the evaluated sentences')

    group.add_argument(
        '--length-penalty',
        type=float,
        default=0.6,
        help='Divides the hypothesis log probabilities in beam search by length^<length penalty>.'
    )

    group.add_argument(
        '--beam-width',
        default=4,
        type=int,
        help='Default beam width for beam search decoder.'
    )

    group.add_argument(
        '--beam-search-all',
        action='store_true',
        help='Default beam width for beam search decoder.'
    )

    return group


def add_data_args(parser):
    group = parser.add_argument_group('Data')

    group.add_argument('--max-length', action='store', type=int, default=51,
                            help='Specify the max length of a sentence')

    group.add_argument('--span-size', action='store', type=int, default=3,
                            help='Specify the span size of the model')

    group.add_argument('--seed', action='store', type=int, default=None,
                            help='Set seed for random scheduler')

    group.add_argument('--shuffle', action='store', type=bool, default=True,
                            help='Shuffle the dataloader')

    group.add_argument('--dataset', action='store', type=str, default="WMT",
                            help='Specify which data to use')

    group.add_argument(
        '--batch-method',
        type=str,
        default='token',
        choices=['token', 'example'],
        help='By which method to sample batches'
    )

    group.add_argument(
        '--batch-size-buffer',
        type=int,
        default=0,
        help='By how many tokens to reduce the batch size on the GPU of the optimizer'
    )

    group.add_argument(
        '--drop-last',
        action='store',
        type=bool,
        default=True,
        help='Whether or not to drop the last minibatch. If it is training, and using multiple GPU, then drop.'
             'If it is evaluating, and using one GPU, then do not drop.'
    )

    return group


def add_cuda_args(parser):
    group = parser.add_argument_group('CUDA')

    group.add_argument(
        '--profile-cuda-memory',
        default=False,
        const='cuda.prof',
        nargs='?',
        type=str,
        help='Whether to profile CUDA memory.'
    )

    return group


def get_cl_args():
    """Get the command line arguments using argparse."""
    arg_parser = argparse.ArgumentParser(prog="RNN-NMT-Syntax", description='Train machine translation model with RNN + Syntax')

    arg_parser.add_argument('--experiment-path', action='store', type=str, default='experiments/exptest/',
                            help='Specify the path to store the experiment')

    arg_parser.add_argument('-s', '--save', action='store', type=str, default='checkpoint.pth.tar',
                            help='Specify the path of checkpoint to save the stored model')

    arg_parser.add_argument('--save-loss-every', action='store', type=int, default=10,
                            help='Save loss every x steps')

    arg_parser.add_argument('--save-checkpoint-every', action='store', type=int, default=50,
                            help='Save checkpoint every x steps')

    arg_parser.add_argument('-b', '--best-model', action='store', type=str, default='model_best.pth.tar',
                            help='Specify the path of checkpoint to save the best stored model')

    arg_parser.add_argument('-r', '--restore', action='store', type=str, default=None,
                            help='Specify the path of checkpoint to load the stored model')

    arg_parser.add_argument('--track', action='store_true',
                            help='Track this run in experiment')

    arg_parser.add_argument('--mode', action='store', type=str, default="train",
                            help='Specify train or evaluate, if evaluate, need to load a model')

    groups = {}
    groups['lstm'] = add_lstm_args(arg_parser)
    groups['data'] = add_data_args(arg_parser)
    groups['train'] = add_train_args(arg_parser)
    groups['evaluate'] = add_evaluate_args(arg_parser)
    groups['cuda'] = add_cuda_args(arg_parser)


    return arg_parser.parse_args()
