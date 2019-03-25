import torch
import random
import time
import numpy as np
from model import DEVICE, SOS_token, EOS_token

# config: max_length, span_size, hidden_size

class Evaluator(object):
    def __init__(self, config, models, dataset, experiment=None):
        self.config = config
        self.encoder = models['encoder']
        self.decoder = models['decoder']
        self.dataset = dataset
        self.experiment = experiment

    def translate_batch(self, batch):
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            batch_size = len(batch)
            input_tensors =[self.dataset.tensor_from_sentence(sentence) for sentence in batch]
            # input_tensors = sorted(input_tensors, key=lambda x: x.size()[0], reverse=True)
            input_lengths_list = [x.size()[0] for x in input_tensors]
            input_lengths_np = np.array(input_lengths_list)
            input_lengths_np_order = np.argsort(input_lengths_np)[::-1]
            input_lengths_np_order_order = np.argsort(input_lengths_np_order)
            # print("input_tensors", input_tensors)
            # print(input_lengths_np_order)
            # print(type(input_lengths_np_order))
            # print([input_lengths_list[i] for i in input_lengths_np_order])
            # print("i", [input_tensors[i] for i in input_lengths_np_order])
            input_lengths = torch.LongTensor([input_lengths_list[i] for i in input_lengths_np_order], device=torch.device("cpu"))

            input_batches = torch.nn.utils.rnn.pad_sequence([input_tensors[i] for i in input_lengths_np_order], batch_first=True)
            encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths)
            # encoder_outputs2 = torch.zeros((batch_size, self.config['max_length'], self.config['hidden_size']),
            #                                dtype=torch.float, device=DEVICE)
            # encoder_outputs2[:, :encoder_outputs.size()[1]] += encoder_outputs
            span_seq_len =  int(self.config['max_length']/self.config['span_size'])

            decoder_input = torch.tensor([SOS_token] * self.config['span_size'] * batch_size, device=DEVICE).view(batch_size, -1)
            decoder_outputs = torch.zeros((batch_size, self.config['max_length']), dtype=torch.long, device=DEVICE)

            decoder_hidden = encoder_hidden
            for l in range(span_seq_len):
                decoder_output, decoder_hidden, decoder_attn = self.decoder(decoder_input, decoder_hidden,
                                                                            encoder_outputs)
                topv, topi = decoder_output.topk(1, dim=2)
                # print("topi", topi.size())
                decoder_input = topi
                # print("decoder_outputs", decoder_outputs.size())
                # print("decoder_outputs[:, l:l+self.config['span_size']]", decoder_outputs[:, l:l+self.config['span_size']].size())
                # print("topi", topi.size())
                decoder_outputs[:, l:l+self.config['span_size']] = topi.squeeze(2)
            # print("decoder_outputs", decoder_outputs.size())
            decoded_sentences_sorted = [[self.dataset.index2word[w.item()] for w in tensor_sentence]
                                        for tensor_sentence in decoder_outputs]
            print(decoded_sentences_sorted)
            decoded_words = [decoded_sentences_sorted[i] for i in input_lengths_np_order_order]
            self.encoder.train()
            self.decoder.train()
            return decoded_words


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

    def evaluate_randomly(self, dataset_split='valid', evaluate_size=None):
        if evaluate_size is not None:
            evaluating_pairs = self.dataset.pairs[dataset_split][:evaluate_size]
        else:
            evaluating_pairs = self.dataset.pairs[dataset_split]
        pairs = random.sample(evaluating_pairs, self.config['num_evaluate'])
        print("pairs", pairs)

        output_words = self.translate_batch([pair[0] for pair in pairs])
        for i in range(self.config['num_evaluate']):
            print('>', pairs[i][0])
            print('=', pairs[i][1])
            output_sentence = ' '.join(output_words[i])
            print('<', output_sentence)
            print('')

        # for i in range(self.config['num_evaluate']):
        #     pair = random.choice(self.dataset.pairs[dataset_split])
        #     print('>', pair[0])
        #     print('=', pair[1])
        #     output_words, attentions = self.translate(pair[0])
        #     output_sentence = ' '.join(output_words)
        #     print('<', output_sentence)
        #     print('')

    def evaluate(self, dataset_split='val'):
        pairs = self.dataset.pairs[dataset_split]
        start = time.time()
        preds = self.translate_batch(pairs)
        print("Evaluation time for {} sentences is {}".format(len(pairs), time.time() - start))
        return preds
