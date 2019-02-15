
import torch
import random
from torch import nn, optim
from model import SOS_token, EOS_token, MAX_LENGTH, SPAN_SIZE
from model.utils import *
from model.preprocess2 import tensors_from_pair

# config: max_length, span_size, teacher_forcing_ratio, learning_rate, n_iters


class Trainer(object):
    def __init__(self, config, models, device):
        self.config = config
        self.encoder = models['encoder']
        self.decoder = models['decoder']
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.config['learning_rate'])
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.NLLLoss()
        self.device = device

    def train_iter(self, input_tensor, target_tensor):
        encoder_hidden = self.encoder.init_hidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(self.config['max_length'], self.encoder.hidden_size, device=self.device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = tuple([torch.tensor([[SOS_token]], device=self.device) for i in range(self.config['span_size'])])

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

    def train_epoch(self, epoch, lang, pairs):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        training_pairs = [tensors_from_pair(lang, lang, random.choice(pairs))
                          for i in range(self.config['n_iters'])]
        criterion = nn.NLLLoss()

        best_loss = float("inf")

        checkpoint_loaded = False

        start_iter = 1

        num_exceptions = 0

        for iter in range(start_iter, self.config['n_iters'] + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            # loss = train(input_tensor, target_tensor, encoder,
            #              decoder, encoder_optimizer, decoder_optimizer, criterion, num_layers=num_layers)
            try:
                loss = self.train_iter(input_tensor, target_tensor)
            except:
                num_exceptions += 1
                continue

            # prepare information to print
            if best_loss > loss:
                best_loss = loss
                is_best = True
            else:
                is_best = False
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (time_since(start, iter / self.config['n_iters']),
                                             iter, iter / self.config['n_iters'] * 100, print_loss_avg))
                save_checkpoint({
                    'epoch': iter + 1,
                    'encoder_state': self.encoder.state_dict(),
                    'decoder_state': self.decoder.state_dict(),
                    'loss': loss,
                    'encoder_optimizer': self.encoder_optimizer.state_dict(),
                    'decoder_optimizer': self.decoder_optimizer.state_dict(),
                }, is_best)

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        show_plot(plot_losses)

    def restore(self, restore_path):
        # load checkpoint
        restore_checkpoint(self.encoder, self.decoder, self.encoder_optimizer, self.decoder_optimizer, restore_path)
