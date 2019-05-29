import os
import torch
from torch import nn
import random
import time
import numpy as np
from model import DEVICE, SOS_token, EOS_token
from model.beam_search2 import BeamSearchDecoder, Beam

# config: max_length, span_size, hidden_size


class Evaluator(object):
    def __init__(self, config, models, dataloader, experiment=None):
        self.config = config
        self.encoder = models['encoder']
        self.decoder = models['decoder']
        if 'cuda' in DEVICE.type:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)
        self.encoder.eval()
        self.decoder.eval()
        self.dataloader = dataloader
        self.beam_search_decoder = BeamSearchDecoder(self.decoder, self.config)
        self.experiment = experiment

    @property
    def dataset(self):
        ''' Get the dataset '''
        return self.dataloader.dataset

    @property
    def sos_idx(self):
        ''' Get the sos idx '''
        return self.dataset.sos_idx

    def generate_batch_greedy(self, batch_inputs, batch_input_lens):
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()

            batch_size = len(batch_inputs)
            # print([self.dataloader.dataset.index2word[w.item()] for w in batch_inputs[0]])

            encoder_outputs, encoder_hidden, encoder_cell = self.encoder(batch_inputs.to(device=DEVICE),
                                                                         batch_input_lens,
                                                                         batch_inputs.size()[1])

            span_seq_len = int(self.config['max_length'] / self.config['span_size'])

            decoder_hidden = torch.zeros(self.config['num_layers'] + 1 + self.config['more_decoder_layers'],
                                         batch_inputs.size()[0], self.config['hidden_size'], device=DEVICE)
            decoder_cell = torch.zeros(self.config['num_layers'] + 1 + self.config['more_decoder_layers'],
                                       batch_inputs.size()[0], self.config['hidden_size'], device=DEVICE)
            decoder_input = torch.tensor([SOS_token] * self.config['span_size'] * batch_size, device=DEVICE).view(
                batch_size, -1)
            decoder_outputs = torch.zeros((batch_size, self.config['max_length']), dtype=torch.long, device=DEVICE)

            for i in range(0, span_seq_len * self.config['span_size'], self.config['span_size']):
                decoder_output, decoder_hidden, decoder_cell, decoder_attn = self.decoder(decoder_input,
                                                                            decoder_hidden, decoder_cell, encoder_outputs)
                topv, topi = decoder_output.topk(1, dim=2)
                decoder_input = topi.squeeze(2)
                decoder_outputs[:, i:i + self.config['span_size']] = topi.squeeze(2)

            decoded_words = [[self.dataloader.dataset.index2word[w.item()] for w in tensor_sentence]
                                 for tensor_sentence in decoder_outputs]
            return decoded_words

    def generate_batch_beam(self, batch_inputs, batch_input_lens):
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()

            batch_size = len(batch_inputs)

            encoder_outputs, encoder_hidden, encoder_cell = self.encoder(batch_inputs.to(device=DEVICE),
                                                                         batch_input_lens,
                                                                         batch_inputs.size()[1])

            return self.beam_search_decoder.decode_batch(encoder_outputs, encoder_hidden,
                                                         torch.tensor([[self.sos_idx] * self.config['span_size'] *
                                                                       batch_size], dtype=torch.long)
                                                         .view(batch_size, -1))

    def evaluate_beam(self):
        batches = self.dataloader
        start = time.time()
        ordered_outputs = []
        for batch in batches:
            beams = self.generate_batch_beam(batch['inputs'], batch['input_lens'])
            print("inputs", [self.dataloader.dataset.index2word[w.item()] for w in batch['inputs'][0]])
            for i, example_id in enumerate(batch['example_ids']):
                outputs = []
                beam = beams[i]
                sequence = beam.best_hypothesis.sequence[self.config['span_size']:]
                decoded = [self.dataloader.dataset.index2word[w.item()] for w in sequence]
                outputs.append(decoded)
                if i == 0:
                    print("example_id", example_id)
                    print("decoded", decoded)
                    if example_id == 789:
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                ordered_outputs.append((example_id, outputs))
        print("Evaluation time for {} sentences is {} for checkpoint {}".format(len(self.dataloader.dataset.pairs),
                                                                                time.time() - start,
                                                                                self.config['restore']))
        preds = []
        for _, outputs in sorted(ordered_outputs, key=lambda x: x[0]):  # pylint:disable=consider-using-enumerate
            preds.extend(outputs)
        return preds

    def evaluate_greedy(self):
        batches = self.dataloader
        start = time.time()
        preds = []
        ordered_outputs = []
        for i, batch in enumerate(batches, 1):
            # print("batch", [self.dataloader.dataset.index2word[w.item()] for w in batch['inputs'][0]])
            pred = self.generate_batch_greedy(batch['inputs'], batch['input_lens'])
            for i, example_id in enumerate(batch['example_ids']):
                ordered_outputs.append((example_id, [pred[i]]))
            # preds.extend(pred)
            print("output", ordered_outputs[0])
        print("Evaluation time for {} sentences is {} for checkpoint {}".format(len(self.dataloader.dataset.pairs),
                                                                                time.time() - start,
                                                                                self.config['restore']))
        for _, outputs in sorted(ordered_outputs, key=lambda x: x[0]):  # pylint:disable=consider-using-enumerate
            preds.extend(outputs)
        return preds

    def evaluate(self, method):
        if method == 'greedy':
            return self.evaluate_greedy()
        elif method == 'beam':
            return self.evaluate_beam()
        else:
            raise ValueError("Unknown evaluate method!!")

    def restore_checkpoint(self, restore_path):
        if restore_path is not None:
            if self.config['average_checkpoints']:
                path = restore_path + str(self.config['start_epoch']) + '.pth.tar'
                if os.path.isfile(path):
                    state = torch.load(path)
                    print("=> loaded checkpoint '{}' (epoch {})".format(path, state['epoch']))
                    models = ['encoder_state', 'decoder_state']
                    model_states = {model: state[model] for model in models}
                    count = 1
                    for epoch in range(self.config['start_epoch'] + 1, self.config['end_epoch'] + 1):
                        path = restore_path + str(epoch) + '.pth.tar'
                        if os.path.isfile(path):
                            checkpoint = torch.load(path)
                            for model in models:
                                new_model_state = checkpoint[model]
                                for name, param in model_states[model].items():
                                    param.mul_(count).add_(new_model_state[name]).div_(count + 1)
                            count += 1
                            print("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
                        else:
                            print("=> no checkpoint found at '{}'".format(path))
                    self.encoder.load_state_dict(model_states['encoder_state'])
                    self.decoder.load_state_dict(model_states['decoder_state'])
                else:
                    print("=> no checkpoint found at '{}'".format(path))
            else:
                if os.path.isfile(restore_path):
                    print("=> loading checkpoint '{}'".format(restore_path))
                    checkpoint = torch.load(restore_path)
                    print("encoder=======")
                    for state in checkpoint['encoder_state']:
                        print(state, checkpoint['encoder_state'][state].shape)
                    print("decoder=======")
                    for state in checkpoint['decoder_state']:
                        print(state, checkpoint['decoder_state'][state].shape)
                    self.encoder.load_state_dict(checkpoint['encoder_state'])
                    self.decoder.load_state_dict(checkpoint['decoder_state'])
                    print("=> loaded checkpoint '{}' (epoch {})".format(restore_path, checkpoint['epoch']))
                else:
                    print("=> no checkpoint found at '{}'".format(restore_path))
