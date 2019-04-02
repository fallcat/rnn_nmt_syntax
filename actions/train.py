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
        self.encoder_lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.encoder_optimizer,
            config['lr_decay']
        )
        self.decoder_lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.decoder_optimizer,
            config['lr_decay']
        )
        self.criterion = nn.NLLLoss(ignore_index=0)
        self.epoch = -1
        self.step = -1
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.experiment = experiment
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
        :return:
        """

        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_length = sum(batch['input_lens']).item() + sum(batch['target_lens']).item()

        # Run words through encoder
        try:
            # print("Batch size:", batch['inputs'].size()[0])
            # print("Total length:", batch['inputs'].size()[1])
            encoder_outputs, encoder_hidden = self.encoder(batch['inputs'].to(device=DEVICE), batch['input_lens'], batch['inputs'].size()[1])
            # except Exception as ex:
            #     template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            #     message = template.format(type(ex).__name__, ex.args)
            #     print(message)
            #     print("Batch size:", batch['inputs'].size()[0])
            #     print("Total length:", batch['inputs'].size()[1])
            #     print(batch['inputs'])

            targets2 = torch.zeros((batch['batch_size'], batch['span_seq_len'] * self.config['span_size']),  dtype=torch.long, device=DEVICE)
            targets2[:, :batch['targets'].size()[1]] = batch['targets']
            decoder_hidden = encoder_hidden
            decoder_outputs = torch.zeros((batch['batch_size'], batch['span_seq_len'] * self.config['span_size'],
                                           self.dataset.num_words), dtype=torch.float, device=DEVICE)
            # print("decoder_outputs.get_device()", decoder_outputs.get_device())
            for i in range(batch['span_seq_len']):
                # print("decoding at ", i)
                decoder_output, decoder_hidden, decoder_attn = self.decoder(targets2[:, i:i+self.config['span_size']],
                                                                            decoder_hidden, encoder_outputs)
                decoder_outputs[:, i:i+self.config['span_size']] = decoder_output

            loss = self.criterion(decoder_outputs[:, :-self.config['span_size']].contiguous().view(-1, self.dataset.num_words),
                                   targets2[:, self.config['span_size']:].contiguous().view(-1))

            # try:
            loss = loss.sum()
            loss.backward()
            self.encoder_lr_scheduler.step()
            self.decoder_lr_scheduler.step()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            return loss.item()/total_length
        except RuntimeError as rte:
            if 'out of memory' in str(rte):
                torch.cuda.empty_cache()
            else:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(rte).__name__, rte.args)
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

        # def get_description():
        #     description = f'Train #{epoch}'
        #     # if verbose > 0:
        #     #     description += f' {self.metric_store}'
        #     # if verbose > 1:
        #     #     description += f' [{profile.mem_stat_string(["allocated"])}]'
        #     return description

        # batches = tqdm.tqdm(
        #     self.dataloader,
        #     unit='batch',
        #     dynamic_ncols=True,
        #     desc=get_description(),
        #     file=sys.stdout  # needed to make tqdm_wrap_stdout work
        # )
        batches = self.dataloader
        len_batches = len(batches)

        accumulated_loss = 0
        accumulated_loss_n = 0

        # with tqdm_wrap_stdout():
        for i, batch in enumerate(batches, 1):

            self.step = i
            if self.experiment is not None:
                self.experiment.set_step(i)
            # loss = self.train_batch3(batch)
            try:
                # start_step = time.time()
                loss = self.train_batch3(batch)
                epoch_loss += loss
                accumulated_loss += loss
                accumulated_loss_n += 1

                if self.experiment is not None and (i % self.config['save_loss_every'] == 0 or i == len_batches):
                    self.experiment.log_metric("loss", accumulated_loss/accumulated_loss_n)
                    # self.experiment.log_metric("learning_rate", self.encoder_optimizer.param_groups['lr'])
                    accumulated_loss = 0
                    accumulated_loss_n = 0
                if i % self.config['save_checkpoint_every'] == 0 or i == len_batches:
                    self.save_checkpoint({
                        'epoch': epoch,
                        'step': i,
                        'encoder_state': self.encoder.state_dict(),
                        'decoder_state': self.decoder.state_dict(),
                        'encoder_optimizer': self.encoder_optimizer.state_dict(),
                        'decoder_optimizer': self.decoder_optimizer.state_dict(),
                        'encoder_lr_scheduler': self.encoder_lr_scheduler.state_dict(),
                        'decoder_lr_scheduler': self.decoder_lr_scheduler.state_dict()
                    })
                # print("time for batch {} is {}".format(i, time.time()-start_step))

            except RuntimeError as rte:
                if 'out of memory' in str(rte):
                    torch.cuda.empty_cache()
                    oom += 1
                    self.experiment.log_metric('oom', oom)
                else:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(rte).__name__, rte.args)
                    print(message)
                    return -1

        print('%s (%d %d%%) %.10f' % (
            time_since(start, (epoch + 1) / self.config['num_epochs']),
            epoch + 1, (epoch + 1) / self.config['num_epochs'] * 100,
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
                    self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
                    self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
                    self.encoder_lr_scheduler.load_state_dict(checkpoint['encoder_lr_scheduler'])
                    self.decoder_lr_scheduler.load_state_dict(checkpoint['decoder_lr_scheduler'])
                except:
                    print("exception when loading state dict to optimizer and lr scheduler")
                print("=> loaded checkpoint '{}' (epoch {}, step {})".format(restore_path, checkpoint['epoch'], checkpoint['step']))
            else:
                print("=> no checkpoint found at '{}'".format(restore_path))

    def save_checkpoint(self, state):
        torch.save(state, self.config['experiment_path'] + self.config['save_path'])
        # if is_best:
        #     shutil.copyfile(self.config['save_path'], self.config['best_save_path'])
