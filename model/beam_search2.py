import torch
from model import utils

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


class BeamSearchDecoder(object):
    def __init__(self, decoder, config):
        self.decoder = decoder
        self.config = config

    def decode(self, encoder_outputs, encoder_hidden, beams):
        self.decoder.eval()
        with torch.no_grad():
            encoded_hidden_list = utils.split_or_chunk((encoder_outputs, encoder_hidden.transpose(0, 1)), len(beams))
            for row in encoded_hidden_list:
                for l in self.config['max_length']:

