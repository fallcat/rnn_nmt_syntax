"""
Main entrance of the program
"""

from comet_ml import Experiment
from model.utils import get_cl_args, save_predictions, get_random_seed_fn
from data.utils import get_dataloader
from data.wmt import WMTDataset
from data.iwslt import IWSLTDataset
from actions.train import Trainer
from actions.evaluate import Evaluator
from model.seq2seq import EncoderRNN, AttnKspanDecoderRNN, BatchEncoderRNN, BatchAttnKspanDecoderRNN3
from model import DEVICE, NUM_DEVICES

# config: max_length, span_size, teacher_forcing_ratio, learning_rate, num_iters, print_every, plot_every, save_path,
#         restore_path, best_save_path, plot_path

def main():
    # max_length needs to be multiples of span_size
    args = get_cl_args()
    print(args)
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
        'evaluate_every': args.evaluate_every,
        'optimizer': args.optimizer,
        'dataset': args.dataset,
        'mode': args.mode,
        'evaluate_path': args.evaluate_path,
        'seed': args.seed,
        'shuffle': args.shuffle
    }

    datasets = {"WMT": WMTDataset, "IWSLT": IWSLTDataset}
    dataset_train = datasets[args.dataset](max_length=args.max_length, span_size=args.span_size, split="train")
    profile_cuda_memory = args.profile_cuda_memory
    pin_memory = 'cuda' in DEVICE.type and not profile_cuda_memory

    if args.seed is not None:
        args.seed_fn = get_random_seed_fn(args.seed)
        args.seed_fn()
    else:
        args.seed_fn = None

    dataloader_train = get_dataloader(
        dataset_train, args.seed_fn, pin_memory,
        NUM_DEVICES, shuffle=args.shuffle
    )
    encoder1 = BatchEncoderRNN(dataset_train.num_words, args.hidden_size, num_layers=args.num_layers).to(DEVICE)
    attn_decoder1 = BatchAttnKspanDecoderRNN3(args.hidden_size, dataset_train.num_words, num_layers=args.num_layers,
                                              dropout_p=args.dropout, max_length=args.max_length,
                                              span_size=args.span_size).to(DEVICE)
    models = {'encoder': encoder1, 'decoder': attn_decoder1}

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

        experiment.log_parameters(config)
    else:
        experiment = None

    trainer = Trainer(config=config, models=models, dataset=dataloader_train, experiment=experiment)
    if args.restore is not None:
        trainer.restore_checkpoint(args.restore)
    if args.mode == "train":
        trainer.train_and_evaluate(args.train_size)
    elif args.mode == "evaluate":
        models = {'encoder': trainer.encoder, 'decoder': trainer.decoder}
        dataset_valid = datasets[args.dataset](max_length=args.max_length, span_size=args.span_size, split="valid")
        evaluator = Evaluator(config=config, models=models, dataset=dataset_valid, experiment=experiment)
        preds = evaluator.evaluate()
        save_predictions(preds, args.evaluate_path)


if __name__ == "__main__":
    main()
