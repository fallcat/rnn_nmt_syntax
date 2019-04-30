"""
Main entrance of the program
"""


from comet_ml import Experiment
import torch
from model.utils import save_predictions, get_random_seed_fn
from args import get_cl_args
from data.utils import get_dataloader
from data.wmt import WMTDataset
from data.iwslt import IWSLTDataset
from actions.train import Trainer
from actions.evaluate import Evaluator
from model.seq2seq import BatchEncoderRNN, BatchAttnKspanDecoderRNN, BatchAttnKspanDecoderRNNSmall, \
    BatchBahdanauAttnKspanDecoderRNN, BatchKspanDecoderRNN, Encoder, Decoder
from model import DEVICE, NUM_DEVICES

# config: max_length, span_size, teacher_forcing_ratio, learning_rate, num_iters, print_every, plot_every, save_path,
#         restore_path, best_save_path, plot_path


def main():
    # max_length needs to be multiples of span_size
    # mp.set_start_method('spawn')
    args = get_cl_args()
    print(args)
    print("Number of GPUs:", torch.cuda.device_count())
    config = {
        'max_length': args.max_length,
        'span_size': args.span_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'print_every': args.print_every,
        'save_path': args.save_path,
        'restore_path': args.restore,
        'best_save_path': args.best_model,
        'minibatch_size': args.minibatch_size,
        'num_epochs': args.num_epochs,
        'num_evaluate': args.num_evaluate,
        'hidden_size': args.hidden_size,
        'optimizer': args.optimizer,
        'dataset': args.dataset,
        'mode': args.mode,
        'evaluate_path': args.evaluate_path,
        'seed': args.seed,
        'shuffle': args.shuffle,
        'batch_size_buffer': args.batch_size_buffer,
        'batch_method': args.batch_method,
        'lr_decay': args.lr_decay,
        'experiment_path': args.experiment_path,
        'save_loss_every': args.save_loss_every,
        'save_checkpoint_every': args.save_checkpoint_every,
        'length_penalty': args.length_penalty,
        'drop_last': args.drop_last,
        'beam_width': args.beam_width,
        'beam_search_all': args.beam_search_all,
        'clip': args.clip,
        'search_method': args.search_method,
        'eval_when_train': args.eval_when_train,
        'filter': args.filter,
        'detokenize': args.detokenize,
        'rnn_type': args.rnn_type,
        'num_layers': args.num_layers,
        'teacher_forcing_ratio': args.teacher_forcing_ratio,
        'num_directions': args.num_directions,
        'trim': args.trim
    }

    # config dataloader

    datasets = {"WMT": WMTDataset, "IWSLT": IWSLTDataset}
    dataset = datasets[args.dataset]
    profile_cuda_memory = args.profile_cuda_memory
    pin_memory = 'cuda' in DEVICE.type and not profile_cuda_memory

    if args.seed is not None:
        args.seed_fn = get_random_seed_fn(args.seed)
        args.seed_fn()
    else:
        args.seed_fn = None

    dataloader_train = get_dataloader(
        dataset, config, "train", args.seed_fn, pin_memory,
        NUM_DEVICES, shuffle=args.shuffle
    )

    dataloader_valid = get_dataloader(
        dataset, config, "valid", args.seed_fn, pin_memory,
        NUM_DEVICES, shuffle=args.shuffle
    )

    # define the models

    torch.cuda.empty_cache()

    encoder1 = BatchEncoderRNN(dataloader_train.dataset.num_words,
                               args.hidden_size,
                               num_layers=args.num_layers,
                               rnn_type=args.rnn_type,
                               num_directions= args.num_directions).to(DEVICE)
    attn_decoder1 = BatchAttnKspanDecoderRNNSmall(args.hidden_size,
                                             dataloader_train.dataset.num_words,
                                             num_layers=args.num_layers,
                                             dropout_p=args.dropout,
                                             max_length=args.max_length,
                                             span_size=args.span_size,
                                             rnn_type=args.rnn_type,
                                             num_directions=args.num_directions).to(DEVICE)
    models = {'encoder': encoder1, 'decoder': attn_decoder1}

    if args.track:
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

    if args.mode == "train":
        trainer = Trainer(config=config, models=models, dataloader=dataloader_train, dataloader_valid=dataloader_valid,
                          experiment=experiment)
        if args.restore is not None:
            trainer.restore_checkpoint(args.experiment_path + args.restore)
        trainer.train()
    elif args.mode == "evaluate":
        evaluator = Evaluator(config=config, models=models, dataloader=dataloader_valid, experiment=experiment)
        if args.restore is not None:
            evaluator.restore_checkpoint(args.experiment_path + args.restore)
        preds = evaluator.evaluate(args.search_method)
        save_predictions(preds, args.evaluate_path, args.detokenize)
    elif args.mode == "evaluate_train":
        evaluator = Evaluator(config=config, models=models, dataloader=dataloader_train, experiment=experiment)
        if args.restore is not None:
            evaluator.restore_checkpoint(args.experiment_path + args.restore)
        preds = evaluator.evaluate(args.search_method)
        save_predictions(preds, args.evaluate_path, args.detokenize)


if __name__ == "__main__":
    main()
