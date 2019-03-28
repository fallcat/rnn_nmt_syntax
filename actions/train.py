import os
import sys
import torch
import random
import time
import tqdm
import shutil
from torch import nn, optim
from torch.autograd import Variable
from model import SOS_token, EOS_token, DEVICE
from model.utils import save_plot, time_since, debug_memory, tqdm_wrap_stdout
from actions.evaluate import Evaluator

# config: max_length, span_size, teacher_forcing_ratio, learning_rate, num_iters, print_every, plot_every, save_path,
#         restore_path, best_save_path, plot_path, minibatch_size, optimizer


class Trainer(object):
    def __init__(self, config, models, dataloader, experiment=None):
        self.config = config
        self.encoder = models['encoder']
        self.decoder = models['decoder']
        optimizers = {"SGD": optim.SGD, "Adadelta": optim.Adadelta, "Adagrad": optim.Adagrad, "RMSprop": optim.RMSprop, "Adam": optim.Adam}
        self.encoder_optimizer = optimizers[self.config["optimizer"]](self.encoder.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        self.decoder_optimizer = optimizers[self.config["optimizer"]](self.decoder.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        self.criterion = nn.NLLLoss(ignore_index=0)
        self.epoch = -1
        self.step = -1
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.experiment = experiment
        self.metric_store = {'oom': 0}

    def train_batch(self, training_pairs):
        """
        train a batch of tensors
        :param training_pairs: list of tensors
        :return:
        """

        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss = 0  # Added onto for each word
        # debug_memory()
        # try:
        #     print("memory allocated", torch.cuda.memory_allocated())
        #     print("memory cached", torch.cuda.memory_cached())
        # except:
        #     pass

        # sort input tensors by length
        batches = sorted(training_pairs, key=lambda x: x[0].size()[0], reverse=True)
        input_list = [x[0] for x in batches]
        # print("inp [0]", input_list[0])
        output_list = [x[1] for x in batches]
        input_lengths = torch.LongTensor([x.size()[0] for x in input_list], device=torch.device("cpu"))
        total_length = sum([len(x[0]) + len(x[1]) for x in batches])
        # print("input lengths", input_lengths)
        # output_lengths = [x.size()[0] for x in batches[:][1]]
        # input_batches = sorted(input_tensors, key=lambda x: x.size()[0], reverse=True)
        # input_lengths = [x.size()[0] for x in input_batches]
        batch_size = len(batches)

        input_batches = Variable(torch.nn.utils.rnn.pad_sequence(input_list, batch_first=True))

        # print("input_batches size", input_batches.size())
        decoder_input = torch.tensor([SOS_token] * self.config['span_size'], device=DEVICE)
        output_to_pad = [torch.cat((decoder_input, output_batch), 0) for output_batch in output_list]
        # print(max(output_to_pad, key=lambda x: x.size()))
        # print(type(max(output_to_pad, key=lambda x: x.size())))
        span_seq_len = int((max(output_to_pad, key=lambda x: x.size()).size()[0] - 1)/ self.config['span_size']) + 1
        output_batches = Variable(torch.zeros((batch_size, span_seq_len * self.config['span_size']), dtype=torch.long, device=DEVICE))
        output_batches2 = torch.nn.utils.rnn.pad_sequence(output_to_pad, batch_first=True)
        # print("output_batches size", output_batches.size())
        # print("output_batches2 size", output_batches2.size())
        output_batches[:, :output_batches2.size()[1]] += output_batches2

        # Run words through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths)
        # encoder_outputs2 = torch.zeros((batch_size, self.config['max_length'], self.config['hidden_size']),
        #                                dtype=torch.float, device=DEVICE)
        # print("encoder_outputs2.get_device()", encoder_outputs2.get_device())
        # encoder_outputs2[:, :encoder_outputs.size()[1]] += encoder_outputs
        # print("encoder_outputs2", encoder_outputs2.size())
        # print("encoder_hidden", encoder_hidden.size())
        # span_seq_len = int(self.config['max_length']/self.config['span_size'])
        decoder_hidden = encoder_hidden
        decoder_outputs = torch.zeros((batch_size, span_seq_len * self.config['span_size'], self.dataset.num_words), dtype=torch.float, device=DEVICE)
        # print("decoder_outputs.get_device()", decoder_outputs.get_device())
        for i in range(span_seq_len):
            decoder_output, decoder_hidden, decoder_attn = self.decoder(output_batches[:, i:i+self.config['span_size']],
                                                                        decoder_hidden, encoder_outputs)
            decoder_outputs[:, i:i+self.config['span_size']] = decoder_output
        # debug_memory()
        # try:
        #     print("memory allocated", torch.cuda.memory_allocated())
        #     print("memory cached", torch.cuda.memory_cached())
        # except:
        #     pass

        # print("outside")
        # print("decoder_outputs", decoder_outputs.size())
        # print("output_batches", output_batches.size())

        # print("decoder_outputs[:, :-self.config['span_size']]", decoder_outputs[:, :-self.config['span_size']].size())
        # print("output_batches[:, self.config['span_size']:]", output_batches[:, self.config['span_size']:].size())
        loss += self.criterion(decoder_outputs[:, :-self.config['span_size']].contiguous().view(-1, self.dataset.num_words),
                               output_batches[:, self.config['span_size']:].contiguous().view(-1))

        try:
            loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            return loss.item() / total_length

        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            return -1

    def train_batch3(self, batch):
        """
        train a batch of tensors
        :param batch: batch of sentences
        :return:
        """

        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss = 0  # Added onto for each word
        total_length = sum(batch['input_lens']).item() + sum(batch['target_lens']).item()
        # debug_memory()
        # try:
        #     print("memory allocated", torch.cuda.memory_allocated())
        #     print("memory cached", torch.cuda.memory_cached())
        # except:
        #     pass

        # inputs
        # 'inputs': inputs,
        # 'input_lens': input_lens,
        # 'targets': targets2,
        # 'target_lens': target_lens,
        # 'example_ids': example_ids

        # sort input tensors by length



        # Run words through encoder
        encoder_outputs, encoder_hidden = self.encoder(batch['inputs'], batch['input_lens'])
        # encoder_outputs2 = torch.zeros((batch_size, self.config['max_length'], self.config['hidden_size']),
        #                                dtype=torch.float, device=DEVICE)
        # print("encoder_outputs2.get_device()", encoder_outputs2.get_device())
        # encoder_outputs2[:, :encoder_outputs.size()[1]] += encoder_outputs
        # print("encoder_outputs2", encoder_outputs2.size())
        # print("encoder_hidden", encoder_hidden.size())
        # span_seq_len = int(self.config['max_length']/self.config['span_size'])
        decoder_hidden = encoder_hidden
        decoder_outputs = torch.zeros((batch['batch_size'], batch['span_seq_len'] * self.config['span_size'],
                                       self.dataset.num_words), dtype=torch.float, device=DEVICE)
        # print("decoder_outputs.get_device()", decoder_outputs.get_device())
        for i in range(batch['span_seq_len']):
            decoder_output, decoder_hidden, decoder_attn = self.decoder(batch['target'][:, i:i+self.config['span_size']],
                                                                        decoder_hidden, encoder_outputs)
            decoder_outputs[:, i:i+self.config['span_size']] = decoder_output
        # debug_memory()
        # try:
        #     print("memory allocated", torch.cuda.memory_allocated())
        #     print("memory cached", torch.cuda.memory_cached())
        # except:
        #     pass

        # print("outside")
        # print("decoder_outputs", decoder_outputs.size())
        # print("output_batches", output_batches.size())

        # print("decoder_outputs[:, :-self.config['span_size']]", decoder_outputs[:, :-self.config['span_size']].size())
        # print("output_batches[:, self.config['span_size']:]", output_batches[:, self.config['span_size']:].size())
        loss += self.criterion(decoder_outputs[:, :-self.config['span_size']].contiguous().view(-1, self.dataset.num_words),
                               batch['target'][:, self.config['span_size']:].contiguous().view(-1))

        try:
            loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            return loss.item() / total_length

        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            return -1

    def train_epoch(self, epoch, train_size=None):
        print("===== epoch " + str(epoch) + " =====")
        if self.experiment is not None:
            self.experiment.log_current_epoch(epoch)
        start = time.time()
        epoch_loss = 0
        oom = self.metric_store['oom']

        if train_size is not None:
            pairs = self.dataset.pairs[:train_size]
        else:
            pairs = self.dataset.pairs
        random.shuffle(pairs)

        def get_description():
            description = f'Train #{epoch}'
            # if verbose > 0:
            #     description += f' {self.metric_store}'
            # if verbose > 1:
            #     description += f' [{profile.mem_stat_string(["allocated"])}]'
            return description

        batches = tqdm.tqdm(
            self.dataloader,
            unit='batch',
            dynamic_ncols=True,
            desc=get_description(),
            file=sys.stdout  # needed to make tqdm_wrap_stdout work
        )

        with tqdm_wrap_stdout():
            for i, batch in enumerate(batches, 1):
                self.experiment.set_step(self.experiment.curr_step + 1)
                try:
                    loss = self.train_batch3(batch)
                    epoch_loss += loss

                    if loss != -1:
                        if self.experiment is not None:
                            self.experiment.log_metric("loss", loss)
                        self.save_checkpoint({
                            'epoch': epoch,
                            'step': i,
                            'encoder_state': self.encoder.state_dict(),
                            'decoder_state': self.decoder.state_dict(),
                            'encoder_optimizer': self.encoder_optimizer.state_dict(),
                            'decoder_optimizer': self.decoder_optimizer.state_dict(),
                        })

                except RuntimeError as rte:
                    if 'out of memory' in str(rte):
                        torch.cuda.empty_cache()
                        oom += 1
                        experiment.log_metric('oom', oom)
                    else:
                        batches.close()
                        raise rte


        # for step in range(self.step + 1, int((len(pairs)-1)/self.config['minibatch_size'])+1):
        #
        #     training_pairs_str = [pair for pair in pairs[step * self.config['minibatch_size']:
        #                                                  (step + 1) * self.config['minibatch_size']]]
        #     training_pairs = [self.dataset.tensors_from_pair(pair) for pair in training_pairs_str]
        #
        #     # best_loss = float("inf")
        #
        #     num_exceptions = 0
        #
        #     len_training_pairs = len(training_pairs)
        #
        #     step_loss = 0
        #     step_loss_count = 0
        #
        #     # train batch
        #     loss = self.train_batch(training_pairs)
        #     epoch_loss += loss
        #
        #     # Log to Comet.ml
        #     if loss != -1:
        #         if self.experiment is not None:
        #             self.experiment.log_metric("loss", loss, step=step)
        #         self.save_checkpoint({
        #             'epoch': epoch,
        #             'step': step,
        #             'encoder_state': self.encoder.state_dict(),
        #             'decoder_state': self.decoder.state_dict(),
        #             'encoder_optimizer': self.encoder_optimizer.state_dict(),
        #             'decoder_optimizer': self.decoder_optimizer.state_dict(),
        #         })
        #
        #
        #     if num_exceptions > 0:
        #         print("Step %s, Number of exceptions: %s" % (step, num_exceptions), flush=True)
        print('%s (%d %d%%) %.10f' % (
            time_since(start, epoch + 1 / self.config['num_epochs']),
            epoch + 1, epoch + 1 / self.config['num_epochs'] * 100,
            epoch_loss), flush=True)
        self.step = -1

    def train(self, train_size=None):
        # dataloader = self.prepare_dataloader(train_size)
        if self.step > -1:
            for epoch in range(self.epoch, self.config['num_epochs']):
                self.train_epoch(epoch, train_size)
        else:
            for epoch in range(self.epoch + 1, self.config['num_epochs']):
                self.train_epoch(epoch, train_size)

    def train_and_evaluate(self, train_size=None):

        if self.step > -1:
            for epoch in range(self.epoch, self.config['num_epochs']):
                self.train_epoch(epoch, train_size)
                if (epoch + 1) % self.config['evaluate_every']  == 0:
                    models = {
                        'encoder': self.encoder,
                        'decoder': self.decoder
                    }
                    evaluator = Evaluator(config=self.config, models=models, dataset=self.dataset, experiment=self.experiment)
                    evaluator.evaluate_randomly(dataset_split='train', evaluate_size=train_size)
        else:
            for epoch in range(self.epoch + 1, self.config['num_epochs']):
                self.train_epoch(epoch, train_size)
                if (epoch + 1) % self.config['evaluate_every'] == 0:
                    models = {
                        'encoder': self.encoder,
                        'decoder': self.decoder
                    }
                    evaluator = Evaluator(config=self.config, models=models, dataset=self.dataset,
                                          experiment=self.experiment)
                    evaluator.evaluate_randomly(dataset_split='train', evaluate_size=train_size)

    def restore_checkpoint(self, restore_path):
        if restore_path is not None:
            if os.path.isfile(restore_path):
                print("=> loading checkpoint '{}'".format(restore_path))
                checkpoint = torch.load(restore_path)
                self.epoch = checkpoint['epoch']
                self.step = checkpoint['step']
                self.encoder.load_state_dict(checkpoint['encoder_state'])
                self.decoder.load_state_dict(checkpoint['decoder_state'])
                self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
                self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
                print("=> loaded checkpoint '{}' (iter {})".format(restore_path, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(restore_path))

    def save_checkpoint(self, state):
        torch.save(state, self.config['save_path'])
        # if is_best:
        #     shutil.copyfile(self.config['save_path'], self.config['best_save_path'])
