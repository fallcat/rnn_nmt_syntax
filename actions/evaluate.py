import torch
import random
from model import DEVICE, SOS_token, EOS_token

# config: max_length, span_size

class Evaluator(object):
    def __init__(self, config, models, dataset, experiment=None):
        self.config = config
        self.encoder = models['encoder']
        self.decoder = models['decoder']
        self.dataset = dataset
        self.experiment = experiment

    def translate(self, sentence):
        with torch.no_grad():
            input_tensor = self.dataset.tensor_from_sentence(sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.init_hidden()

            encoder_outputs = torch.zeros(self.config['max_length'], self.encoder.hidden_size, device=DEVICE)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = tuple([torch.tensor([[SOS_token]], device=DEVICE) for i in range(self.config['span_size'])])  # SOS

            decoder_hidden = encoder_hidden  # [encoder_hidden for _ in range(num_layers)]

            decoded_words = []
            decoder_attentions = torch.zeros(self.config['max_length'], self.config['max_length'])

            break_out = False
            for di in range(int((self.config['max_length'] + 1) / self.config['span_size'])):
                #             print("input[0]", decoder_input[0].shape)
                #             print("hidden[0]", decoder_hidden[0].shape)
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                #             topv, topi = decoder_output.topk(1, dim=1)
                topi = [EOS_token] * self.config['span_size']
                for si in range(self.config['span_size']):
                    topv, topi[si] = decoder_output[si].topk(1)
                decoder_input = tuple([topi[si].squeeze().detach() for si in range(self.config['span_size'])])

                #             decoder_input = tuple(topi.squeeze().detach())
                for si in range(self.config['span_size']):
                    if di * self.config['span_size'] + si < self.config['max_length'] and topi[si].item() != EOS_token:
                        decoded_words.append(self.dataset.index2word[topi[si].item()])
                    else:
                        decoded_words.append('<EOS>')
                        break_out = True
                        break
                if break_out:
                    break

            return decoded_words, decoder_attentions[:di + 1]

    def evaluate_randomly(self, dataset_split='valid'):
        for i in range(self.config['num_evaluate']):
            pair = random.choice(self.dataset.pairs[dataset_split])
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = self.translate(pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')

    def evaluate(self, dataset_split='val'):
        for pair in self.dataset.pairs[dataset_split]:
            pass

