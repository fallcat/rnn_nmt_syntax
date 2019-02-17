"""
Main entrance of the program
"""

from model.utils import get_cl_args
from data.wmt import WMTDataset
from actions.train import Trainer
from model.seq2seq import EncoderRNN, AttnKspanDecoderRNN
from model import DEVICE

# config: max_length, span_size, teacher_forcing_ratio, learning_rate, num_iters, print_every, plot_every, save_path,
#         restore_path, best_save_path, plot_path

def main():
    args = get_cl_args()
    print(args)
    dataset = WMTDataset(max_length=args.max_length)
    encoder1 = EncoderRNN(dataset.num_words, args.hidden_size, num_layers=args.num_layers).to(DEVICE)
    attn_decoder1 = AttnKspanDecoderRNN(args.hidden_size, dataset.num_words, num_layers=args.num_layers,
                                        dropout_p=args.dropout).to(DEVICE)
    models = {'encoder': encoder1, 'decoder': attn_decoder1}
    config = {
        'max_length': args.max_length,
        'span_size': args.span_size,
        'teacher_forcing_ratio': args.teacher_forcing_ratio,
        'learning_rate': args.learning_rate,
        'num_iters': args.num_iters,
        'print_every': args.print_every,
        'plot_every': args.plot_every,
        'save_path': args.save,
        'restore_path': args.restore,
        'best_save_path': args.best_model,
        'plot_path': args.plot,
        'minibatch_size': args.minibatch_size
    }
    trainer = Trainer(config=config, models=models, dataset=dataset)
    if args.restore is not None:
        trainer.restore_checkpoint(args.restore)
    trainer.train(args.train_size)
    # for epoch in range(args.num_epochs):
    #     trainer.train_epoch(epoch, args.train_size)


if __name__ == "__main__":
    main()
