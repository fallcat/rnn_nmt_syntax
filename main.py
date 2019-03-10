"""
Main entrance of the program
"""

from comet_ml import Experiment
from model.utils import get_cl_args
from data.wmt import WMTDataset
from actions.train import Trainer
from actions.evaluate import Evaluator
from model.seq2seq import EncoderRNN, AttnKspanDecoderRNN, BatchEncoderRNN, BatchAttnKspanDecoderRNN2
from model import DEVICE

# config: max_length, span_size, teacher_forcing_ratio, learning_rate, num_iters, print_every, plot_every, save_path,
#         restore_path, best_save_path, plot_path

def main():
    # max_length needs to be multiples of span_size
    args = get_cl_args()
    print(args)
    dataset = WMTDataset(max_length=args.max_length, span_size=args.span_size)
    encoder1 = BatchEncoderRNN(dataset.num_words, args.hidden_size, num_layers=args.num_layers).to(DEVICE)
    attn_decoder1 = BatchAttnKspanDecoderRNN2(args.hidden_size, dataset.num_words, num_layers=args.num_layers,
                                              dropout_p=args.dropout, max_length=args.max_length,
                                              span_size=args.span_size).to(DEVICE)
    models = {'encoder': encoder1, 'decoder': attn_decoder1}
    config = {
        'max_length': args.max_length,
        'span_size': args.span_size,
        'teacher_forcing_ratio': args.teacher_forcing_ratio,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_iters': args.num_iters,
        'print_every': args.print_every,
        'plot_every': args.plot_every,
        'save_path': args.save,
        'restore_path': args.restore,
        'best_save_path': args.best_model,
        'plot_path': args.plot,
        'minibatch_size': args.minibatch_size,
        'num_epochs': args.num_epochs,
        'num_evaluate': args.num_evaluate,
        'hidden_size': args.hidden_size,
        'evaluate_every': args.evaluate_every
    }
    if args.do_experiment:
        experiment = Experiment(project_name="rnn-nmt-syntax",
                                workspace="umass-nlp",
                                auto_metric_logging=False,
                                auto_output_logging=None,
                                auto_param_logging=False,
                                log_git_metadata=False,
                                log_git_patch=False,
                                log_env_details=False,
                                log_graph=False,
                                log_code=False,
                                parse_args=False,)
        hyper_params = {
            'max_length': args.max_length,
            'span_size': args.span_size,
            'teacher_forcing_ratio': args.teacher_forcing_ratio,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'num_iters': args.num_iters,
            'save_path': args.save,
            'restore_path': args.restore,
            'best_save_path': args.best_model,
            'plot_path': args.plot,
            'minibatch_size': args.minibatch_size,
            'num_epochs': args.num_epochs,
            'train_size': args.train_size,
            'hidden_size': args.hidden_size
        }
        experiment.log_parameters(hyper_params)
    else:
        experiment = None
    trainer = Trainer(config=config, models=models, dataset=dataset, experiment=experiment)
    if args.restore is not None:
        trainer.restore_checkpoint(args.restore)
    trainer.train_and_evaluate(args.train_size)
    # evaluator = Evaluator(config=config, models=models, dataset=dataset, experiment=experiment)
    # evaluator.evaluate_randomly()
    # for epoch in range(args.num_epochs):
    #     trainer.train_epoch(epoch, args.train_size)


if __name__ == "__main__":
    main()
