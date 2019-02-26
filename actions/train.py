import os
import torch
import random
import time
import shutil
from torch import nn, optim
from model import SOS_token, EOS_token, DEVICE
from model.utils import save_plot, time_since

# config: max_length, span_size, teacher_forcing_ratio, learning_rate, num_iters, print_every, plot_every, save_path,
#         restore_path, best_save_path, plot_path, minibatch_size


class Trainer(object):
    def __init__(self, config, models, dataset, experiment=None):
        self.config = config
        self.encoder = models['encoder']
        self.decoder = models['decoder']
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.config['learning_rate'])
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.NLLLoss()
        self.epoch = -1
        self.step = -1
        self.dataset = dataset
        self.experiment = experiment

    def train_iter(self, input_tensor, target_tensor):
        encoder_hidden = self.encoder.init_hidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(self.config['max_length'], self.encoder.hidden_size, device=DEVICE)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = tuple([torch.tensor([[SOS_token]], device=DEVICE) for i in range(self.config['span_size'])])

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.config['teacher_forcing_ratio'] else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            break_out = False
            for di in range(int((target_length + 1) / self.config['span_size'])):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_input = []
                for si in range(self.config['span_size']):
                    if di * self.config['span_size'] + si < target_length:
                        loss += self.criterion(decoder_output[si], target_tensor[di * self.config['span_size'] + si])
                        decoder_input.append(target_tensor[di * self.config['span_size'] + si])
                    else:
                        break_out = True
                        break
                if break_out:
                    break
                decoder_input = tuple(decoder_input)

        else:
            # Without teacher forcing: use its own predictions as the next input
            break_out = False
            for di in range(int((target_length + 1) / 2)):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topi = [EOS_token] * self.config['span_size']
                for si in range(self.config['span_size']):
                    topv, topi[si] = decoder_output[si].topk(1)
                decoder_input = tuple(
                    [topi[si].squeeze().detach() for si in range(self.config['span_size'])])  # detach from history as input

                for si in range(self.config['span_size']):
                    if di * self.config['span_size'] + si < target_length and decoder_input[si].item() != EOS_token:
                        loss += self.criterion(decoder_output[si], target_tensor[di * self.config['span_size'] + si])
                    else:
                        break_out = True
                        break
                if break_out:
                    break

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length

    def train_epoch(self, epoch, train_size=None):
        print("===== epoch " + str(epoch) + " =====")
        if self.experiment is not None:
            self.experiment.log_current_epoch(epoch)
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        print_count = 0
        plot_loss_total = 0  # Reset every plot_every
        plot_count = 0

        if train_size is not None:
            pairs = self.dataset.pairs['train'][:train_size]
        else:
            pairs = self.dataset.pairs['train']
        random.shuffle(pairs)

        for step in range(self.step + 1, int((len(pairs)-1)/self.config['minibatch_size'])+1):
            # training_pairs = [self.dataset.tensors_from_pair(pair)
            #                   for pair in pairs[step * self.config['minibatch_size']:
            #                                     (step + 1) * self.config['minibatch_size']]]

            training_pairs_str = [pair for pair in pairs[step * self.config['minibatch_size']:
                                                         (step + 1) * self.config['minibatch_size']]]
            training_pairs = [self.dataset.tensors_from_pair(pair) for pair in training_pairs_str]

            # best_loss = float("inf")

            num_exceptions = 0

            len_training_pairs = len(training_pairs)

            step_loss = 0
            step_loss_count = 0

            for iter in range(1, len_training_pairs + 1):
                training_pair = training_pairs[iter - 1]
                input_tensor = training_pair[0]
                target_tensor = training_pair[1]
                try:
                    loss = self.train_iter(input_tensor, target_tensor)
                    if loss > 10:
                        print("large loss", pairs[step * self.config['minibatch_size'] + iter - 1], "loss", loss)
                    step_loss += loss
                    step_loss_count += 1
                except:
                    num_exceptions += 1
                    print("exception sentences", training_pairs_str[iter - 1])
                    continue

                # prepare information to print
                # if best_loss > loss:
                #     best_loss = loss
                #     is_best = True
                # else:
                #     is_best = False
                print_loss_total += loss
                print_count += 1
                plot_loss_total += loss
                plot_count += 1

            # Log to Comet.ml
            if self.experiment is not None:
                if step_loss_count != 0:
                    step_loss_avg = step_loss / step_loss_count
                    self.experiment.log_metric("loss", step_loss_avg, step=step)
                    print('%s (%d %d%%) %.4f' % (
                        time_since(start, step + 1 / (int((len(pairs) - 1) / self.config['minibatch_size']) + 2)),
                        step + 1, step + 1 / (int((len(pairs) - 1) / self.config['minibatch_size']) + 2) * 100,
                        step_loss_avg), flush=True)
                    self.save_checkpoint({
                        'epoch': epoch,
                        'step': step,
                        'encoder_state': self.encoder.state_dict(),
                        'decoder_state': self.decoder.state_dict(),
                        'encoder_optimizer': self.encoder_optimizer.state_dict(),
                        'decoder_optimizer': self.decoder_optimizer.state_dict(),
                    })
            #
            # if step+1 % self.config['print_every'] == 0:
            #     print_loss_avg = print_loss_total / print_count
            #     print_loss_total = 0
            #     print_count = 0
            #     try:
            #         print('%s (%d %d%%) %.4f' % (time_since(start, step+1 / (int((len(pairs)-1)/self.config['minibatch_size'])+2)),
            #               step+1, step+1 / (int((len(pairs)-1)/self.config['minibatch_size'])+2) * 100, print_loss_avg))
            #         self.save_checkpoint({
            #             'epoch': epoch,
            #             'encoder_state': self.encoder.state_dict(),
            #             'decoder_state': self.decoder.state_dict(),
            #             'encoder_optimizer': self.encoder_optimizer.state_dict(),
            #             'decoder_optimizer': self.decoder_optimizer.state_dict(),
            #         })
            #     except ZeroDivisionError:
            #         print("divide by zero when printing loss")
            #
            # try:
            #     if step+1 % self.config['plot_every'] == 0:
            #         plot_loss_avg = plot_loss_total / plot_count
            #         plot_losses.append(plot_loss_avg)
            #         plot_loss_total = 0
            #         plot_count = 0
            # except ZeroDivisionError:
            #     print("divide by zero when plotting loss")

            if num_exceptions > 0:
                print("Step %s, Number of exceptions: %s" % (step, num_exceptions), flush=True)
        self.step = -1

    def train(self, train_size=None):
        # dataloader = self.prepare_dataloader(train_size)
        for epoch in range(self.epoch + 1, self.config['num_epochs']):
            self.train_epoch(epoch, train_size)

    # def prepare_dataloader(self, train_size):
    #     if train_size is not None:
    #         pairs = self.dataset.pairs[:train_size]
    #     else:
    #         pairs = self.dataset.pairs
    #     pairs = torch.cat([self.dataset.tensors_from_pair(pair) for pair in pairs])
    #     print(pairs)
    #     torch_dataset = Data.TensorDataset(pairs)
    #     loader = Data.DataLoader(
    #         dataset=torch_dataset,
    #         batch_size=self.config['minibatch_size'],
    #         shuffle=True,
    #         num_workers=2
    #     )
    #     return loader

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
