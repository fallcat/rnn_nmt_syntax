import os
import sys
import torch
import random
import time
import tqdm
import shutil
import GPUtil
import psutil
from torch import nn, optim
from torch.autograd import Variable
from model import SOS_token, EOS_token, DEVICE
from model.utils import save_plot, time_since, debug_memory, tqdm_wrap_stdout
from actions.evaluate import Evaluator

# config: max_length, span_size, teacher_forcing_ratio, learning_rate, num_iters, print_every, plot_every, save_path,
#         restore_path, best_save_path, plot_path, minibatch_size, optimizer


class Trainer(object):
    def __init__(self, config, models, dataloader, dataloader_valid=None, experiment=None):
        self.config = config
        self.encoder = models['encoder']
        self.decoder = models['decoder']
        optimizers = {"SGD": optim.SGD, "Adadelta": optim.Adadelta, "Adagrad": optim.Adagrad,
                      "RMSprop": optim.RMSprop, "Adam": optim.Adam}
        self.optimizer = optimizers[self.config['optimizer']](list(self.encoder.parameters()) +
                                                              list(self.decoder.parameters()),
                                                              lr=self.config['learning_rate'],
                                                              weight_decay=self.config['weight_decay'])

        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            config['lr_decay']
        )
        self.criterion = nn.NLLLoss(ignore_index=0)
        self.epoch = -1
        self.step = -1
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.experiment = experiment
        self.dataloader_valid = dataloader_valid
        self.metric_store = {'oom': 0}

        if 'cuda' in DEVICE.type:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)
            self.criterion.should_unsqueeze = True
            self.criterion = nn.DataParallel(self.criterion)

    def train_batch3(self, batch):
        """
        train a batch of tensors
        :param batch: batch of sentences
        :return: float: Average loss per token
        """

        # Zero gradients of both optimizers
        self.optimizer.zero_grad()

        # Run words through encoder
        # Make sure inputs are all gathered to be the longest length of the input, or else error will occur
        total_length = sum(batch['input_lens']).item() + sum(batch['target_lens']).item()
        # print("batch input size", batch['inputs'].size())
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(batch['inputs'], batch['input_lens'], batch['inputs'].size()[1])

        decoder_hidden = encoder_hidden
        decoder_cell = torch.zeros(self.config['num_layers'], batch['inputs'].size()[0], self.config['hidden_size'], device=DEVICE)

        decoder_outputs = []

        use_teacher_forcing = True if random.random() < self.config['teacher_forcing_ratio'] else False
        # print("targets", batch['targets'])
        if use_teacher_forcing:
            for i in range(0, batch['span_seq_len'] * self.config['span_size'], self.config['span_size']):
                decoder_output, decoder_hidden, decoder_cell, decoder_attn = self.decoder(batch['targets'][:, i:i+self.config['span_size']],
                                                                                          decoder_hidden, decoder_cell, encoder_outputs)
                decoder_outputs.append(decoder_output)
            decoder_outputs = torch.cat(decoder_outputs, dim=1)
        else:
            batch_size = len(batch['inputs'])
            decoder_input = torch.tensor([SOS_token] * self.config['span_size'] * batch_size, device=DEVICE).view(batch_size, -1)
            for i in range(0, batch['span_seq_len'] * self.config['span_size'], self.config['span_size']):
                decoder_output, decoder_hidden, decoder_cell, decoder_attn = self.decoder(decoder_input,
                                                                            decoder_hidden, decoder_cell, encoder_outputs)
                topv, topi = decoder_output.topk(1, dim=2)
                # print("topi", topi.size())
                decoder_input = topi.squeeze(2)
                decoder_outputs.append(decoder_output)
            decoder_outputs = torch.cat(decoder_outputs, dim=1)

        # print("decoder_outputs", decoder_outputs.size())
        # print("targets", batch['targets'].size())
        loss = self.criterion(decoder_outputs[:, :-self.config['span_size']].contiguous().view(-1, self.dataset.num_words),
                              batch['targets'][:, self.config['span_size']:].contiguous().view(-1))

        loss = loss #.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config['clip'])
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.config['clip'])
        self.lr_scheduler.step()
        self.optimizer.step()
        return loss.item()

    def train_epoch(self, epoch):
        self.encoder.train()
        self.decoder.train()
        print("===== epoch " + str(epoch) + " =====")
        # print("hihihi")
        if self.experiment is not None:
            self.experiment.log_current_epoch(epoch)
        # print("===== epoch " + str(epoch) + " =====")
        start = time.time()
        epoch_loss = 0
        oom = self.metric_store['oom']

        batches = self.dataloader
        len_batches = len(batches)

        accumulated_loss = 0
        accumulated_loss_n = 0
        # print("begin")

        # with tqdm_wrap_stdout():
        for i, batch in enumerate(batches, 1):
            # print("now in batch", i)

            self.step = i
            if self.experiment is not None:
                self.experiment.set_step(self.experiment.curr_step + 1)
            # loss = self.train_batch3(batch)
            try:
                # print("train now")
                torch.cuda.empty_cache()
                loss = self.train_batch3(batch)
                # GPUtil.showUtilization()
                total_length = sum(batch['input_lens']).item() + sum(batch['target_lens']).item()
                epoch_loss += loss
                accumulated_loss += loss * total_length
                accumulated_loss_n += total_length

                if self.experiment is not None and (i % self.config['save_loss_every'] == 0 or i == len_batches):
                    self.experiment.log_metric("train_nll", accumulated_loss/accumulated_loss_n)
                    # self.experiment.log_metric("learning_rate", self.encoder_optimizer.param_groups['lr'])
                    accumulated_loss = 0
                    accumulated_loss_n = 0
                    vm = psutil.virtual_memory()
                    # print("virtual_memory", vm)
                    vm = dict(vm._asdict())
                    self.experiment.log_metric("available_memory", vm['available'])
                    self.experiment.log_metric("total_memory", vm['total'])
                    self.experiment.log_metric("used_memory", vm['used'])
                    self.experiment.log_metric("free_memory", vm['free'])
                # print("time for batch {} is {}".format(i, time.time()-start_step))

            except RuntimeError as rte:
                if 'out of memory' in str(rte):
                    torch.cuda.empty_cache()
                    oom += 1
                    self.experiment.log_metric('oom', oom)
                    print("Out of memory")
                else:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(rte).__name__, rte.args)
                    print(message)
                    return -1

        print("now save")
        self.save_checkpoint({
            'epoch': epoch,
            'encoder_state': self.encoder.state_dict(),
            'decoder_state': self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict()
            # 'encoder_optimizer': self.encoder_optimizer.state_dict(),
            # 'decoder_optimizer': self.decoder_optimizer.state_dict(),
            # 'encoder_lr_scheduler': self.encoder_lr_scheduler.state_dict(),
            # 'decoder_lr_scheduler': self.decoder_lr_scheduler.state_dict()
        }, epoch)

        print('%s (%d %d%%) %.10f' % (
            time_since(start, (epoch + 1) / self.config['num_epochs']),
            epoch + 1, (epoch + 1) / self.config['num_epochs'] * 100,
            epoch_loss), flush=True)
        self.step = -1

    def train(self):
        if self.experiment is not None:
            self.experiment.set_step(0)
        for epoch in range(self.epoch + 1, self.config['num_epochs']):
            self.train_epoch(epoch)
            if self.config['eval_when_train']:
                self.evaluate_nll()

    def evaluate_nll(self):
        batches = self.dataloader

        accumulated_loss = 0
        accumulated_loss_n = 0

        # with tqdm_wrap_stdout():
        for i, batch in enumerate(batches, 1):
            # loss = self.train_batch3(batch)
            try:
                loss = self.evaluate_nll_batch(batch)
                # GPUtil.showUtilization()
                total_length = sum(batch['input_lens']).item() + sum(batch['target_lens']).item()
                accumulated_loss += loss * total_length
                accumulated_loss_n += total_length

                # print("time for batch {} is {}".format(i, time.time()-start_step))

            except RuntimeError as rte:
                if 'out of memory' in str(rte):
                    torch.cuda.empty_cache()
                    print("Out of memory")
                else:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(rte).__name__, rte.args)
                    print(message)
                    return -1
        valid_nll = accumulated_loss / accumulated_loss_n
        if self.experiment is not None:
            self.experiment.log_metric("valid_nll", valid_nll)
        print("Validation NLL:", valid_nll)

    def evaluate_nll_batch(self, batch):
        """
        train a batch of tensors
        :param batch: batch of sentences
        :return:
        """
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()

            total_length = sum(batch['input_lens']).item() + sum(batch['target_lens']).item()

            # Run words through encoder
            encoder_outputs, encoder_hidden, encoder_cell = self.encoder(batch['inputs'].to(device=DEVICE), batch['input_lens'], batch['inputs'].size()[1])
            # targets2 = torch.zeros((batch['batch_size'], batch['span_seq_len'] * self.config['span_size']),  dtype=torch.long, device=DEVICE)
            # targets2[:, :batch['targets'].size()[1]] = batch['targets']
            decoder_hidden = encoder_hidden
            decoder_cell = torch.zeros(self.config['num_layers'], batch['inputs'].size()[0], self.config['hidden_size'],
                                       device=DEVICE)
            # decoder_outputs = torch.zeros((batch['batch_size'], batch['span_seq_len'] * self.config['span_size'],
            #                                self.dataset.num_words), dtype=torch.float, device=DEVICE)
            decoder_outputs = []
            for i in range(batch['span_seq_len']):
                decoder_output, decoder_hidden, decoder_cell, decoder_attn = self.decoder(batch['targets'][:, i:i+self.config['span_size']],
                                                                            decoder_hidden, decoder_cell, encoder_outputs)
                decoder_outputs.append(decoder_output)
            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            loss = self.criterion(decoder_outputs[:, :-self.config['span_size']].contiguous().view(-1, self.dataset.num_words),
                                   batch['targets'][:, self.config['span_size']:].contiguous().view(-1))

            loss = loss #.sum()
            self.encoder.train()
            self.decoder.train()
            return loss.item()

    def restore_checkpoint(self, restore_path):
        if restore_path is not None:
            if os.path.isfile(restore_path):
                print("=> loading checkpoint '{}'".format(restore_path))
                checkpoint = torch.load(restore_path)
                self.epoch = checkpoint['epoch']
                self.step = checkpoint['step']
                self.encoder.load_state_dict(checkpoint['encoder_state'])
                self.decoder.load_state_dict(checkpoint['decoder_state'])
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                    # self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
                    # self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
                    # self.encoder_lr_scheduler.load_state_dict(checkpoint['encoder_lr_scheduler'])
                    # self.decoder_lr_scheduler.load_state_dict(checkpoint['decoder_lr_scheduler'])
                except:
                    print("exception when loading state dict to optimizer and lr scheduler")
                print("=> loaded checkpoint '{}' (epoch {})".format(restore_path, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(restore_path))

    def save_checkpoint(self, state, epoch):
        torch.save(state, self.config['experiment_path'] + self.config['save_path'] + str(epoch) + ".pth.tar")
        # if is_best:
        #     shutil.copyfile(self.config['save_path'], self.config['best_save_path'])
