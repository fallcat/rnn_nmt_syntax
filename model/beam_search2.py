import torch
from model import utils, DEVICE


class BeamHypothesis(object):
    def __init__(self, sequence, score, hidden):
        self.sequence = sequence
        self.score = score
        self.hidden = hidden

    def __len__(self):
        """ The length of the hypothesis is the length of the sequence """
        return len(self.sequence)


class Beam(object):
    def __init__(self, start_sequence, hidden, initial_score=0, max_length=0, width=4):
        self.width = width
        self.max_length = max_length
        self.hypotheses = [BeamHypothesis(start_sequence, initial_score, hidden)]

    @property
    def best_hypothesis(self):
        """ Returns the current best hypothesis given the score comparator """
        return max(self.hypotheses, key=lambda h: h.score)

    def finished_decoding(self, hypothesis, eos_idx):
        ''' Check if the hypothesis has finished decoding '''
        return eos_idx in hypothesis.sequence or 0 < self.max_length <= len(hypothesis.sequence)

    def collate(self):
        sequences = []
        scores = []
        hiddens = []
        for hypothesis in self.hypotheses:
            sequences.append(hypothesis.sequence)
            scores.append(hypothesis.score)
            hiddens.append(hypothesis.hidden)
        return torch.cat(sequences), scores, torch.cat(hiddens)


class BeamSearchDecoder(object):
    def __init__(self, decoder, sos_idx, config, initial_score=0):
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.config = config
        self.initial_score = initial_score

    def initialize_search(self, start_sequences, max_lengths=0, initial_scores=0, beam_width=4):
        ''' Initialize a batch of beams '''
        beams = []
        if isinstance(max_lengths, int):
            max_lengths = [max_lengths] * len(start_sequences)

        if isinstance(initial_scores, int):
            initial_scores = [initial_scores] * len(start_sequences)

        for sequence, score, max_length in zip(start_sequences, initial_scores, max_lengths):
            beams.append(Beam(sequence, score, max_length, beam_width))

        return beams

    def search_all(self, topv, topi, scores):
        rows, cols = topv.size()
        topv2, topi2 = topv.view(-1).topk(self.config['beam_width'])
        rowi = topi2 // rows
        coli = topi2 - rowi * rows
        for b in range(self.config['beam_width']):
            pass

    def search_sequential(self, topv, topi, scores):
        rows, cols = topv.size()
        topv2, topi2 = topv.view(-1).topk(self.config['beam_width'])
        rowi = topi2 // rows
        coli = topi2 - rowi * rows
        for b in range(self.config['beam_width']):
            topv[b].topk(1)

    def decode(self, encoder_outputs, encoder_hidden, start_sequences):
        self.decoder.eval()
        with torch.no_grad():
            encoded_hidden_list = utils.split_or_chunk((encoder_outputs, encoder_hidden.transpose(0, 1)),
                                                       len(encoder_outputs))
            for i, row in enumerate(encoded_hidden_list):
                beam = Beam(start_sequences[i], row[1].transpose(0, 1), self.initial_score,
                            self.config['max_length'], self.config['beam_width'])
                for l in self.config['max_length']/self.config['span_size']:
                    sequences, scores, hiddens = beam.collate()
                    decoder_output, decoder_hidden, decoder_attn = self.decoder(sequences[:, -self.config['span_size']].to(device=DEVICE),
                                                                                hiddens.transpose(0, 1),
                                                                                row[0])
                    topv, topi = decoder_output.topk(self.config['beam_width'], dim=2)
                    if self.config['beam_search_all']:
                        self.search_all(topv, topi, scores)
                    else:
                        self.search_sequential(topv, topi, scores)

