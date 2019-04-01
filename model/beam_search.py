'''
A module that implements beam search
'''
import numpy as np
import torch

from model import utils

class BeamHypothesis(object):
    ''' A class that represents a particular hypothesis in a beam '''
    def __init__(self, sequence, score):
        self.score = score
        self.sequence = sequence

    def __len__(self):
        ''' The length of the hypothesis is the length of the sequence '''
        return len(self.sequence)


class Beam(object):
    ''' A class that represents a beam in the search '''
    def __init__(self, start_sequence, initial_score=0, max_sequence_length=0, width=4):
        ''' Initialize the beam '''
        self.width = width
        self.max_sequence_length = max_sequence_length
        self.hypotheses = [BeamHypothesis(start_sequence, initial_score)]

    @property
    def best_hypothesis(self):
        ''' Returns the current best hypothesis given the score comparator '''
        return max(self.hypotheses, key=lambda h: h.score)

    def finished_decoding(self, hypothesis, eos_idx):
        ''' Check if the hypothesis has finished decoding '''
        return (
            eos_idx in hypothesis.sequence or
            (
                self.max_sequence_length > 0 and
                len(hypothesis.sequence) >= self.max_sequence_length
            )
        )


class BeamSearchDecoder(object):
    ''' Class that encapsulates decoding using beam search '''
    def __init__(self, model, eos_idx, length_penalty=0, span=1):
        ''' Initialize the beam search decoder '''
        self.span = span
        self.model = model
        self.eos_idx = eos_idx
        self.length_penalty = length_penalty

    def all_done(self, beams):
        ''' Determine if the given beams have completed '''
        return all(
            beam.finished_decoding(hypothesis, self.eos_idx)
            for beam in beams
            for hypothesis in beam.hypotheses
        )

    def collate(self, encoded, beams):
        ''' Collate beams into a batch '''
        batch = []
        beam_map = {}
        encoded_batch = []
        for i, beam in enumerate(beams):
            hypothesis_map = {}
            for hypothesis in beam.hypotheses:
                if beam.finished_decoding(hypothesis, self.eos_idx):
                    continue

                batch_idx = len(batch)
                encoded_batch.append(encoded[i])
                hypothesis_map[hypothesis] = batch_idx
                batch.append(hypothesis.sequence)

            if hypothesis_map:
                beam_map[beam] = hypothesis_map

        batch = torch.LongTensor(batch)
        encoded_batch = utils.cat(encoded_batch)
        return encoded_batch, batch, beam_map

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

    def normalized_score(self, score, length):
        '''
        Calculate the normalized score of the hypothesis

        https://arxiv.org/abs/1609.08144
        See equation #14
        '''
        return score * ((5 + 1) / (5 + length)) ** self.length_penalty

    def update_beams(self, log_prob, beam_map):
        ''' Update the beam batch '''
        for beam, hypothesis_map in beam_map.items():
            for span_idx in range(self.span):
                hypotheses_scores = []
                normalized_hypotheses_scores = []
                for hypothesis in beam.hypotheses:
                    batch_idx = hypothesis_map.get(hypothesis, -1)
                    if batch_idx >= 0:
                        length = len(hypothesis) + 1
                        score = log_prob[batch_idx, :, span_idx] + hypothesis.score
                    else:
                        length = len(hypothesis)
                        score = log_prob.new_full((1,), hypothesis.score)

                    hypotheses_scores.append(score)
                    normalized_hypotheses_scores.append(self.normalized_score(score, length))

                hypothesis_set_lengths = [len(h) for h in hypotheses_scores]
                hypotheses_partition = np.cumsum(hypothesis_set_lengths)

                hypotheses_scores = torch.cat(hypotheses_scores)
                normalized_hypotheses_scores = torch.cat(normalized_hypotheses_scores)

                # pylint:disable=unused-variable
                scores, indices = torch.topk(normalized_hypotheses_scores, beam.width)
                # pylint:enable=unused-variable

                # need to convert each index into a hypothesis index
                # numpy searchsorted is a faster version of python's bisect.bisect[_left|_right]
                # that returns insertion points for multiple values
                new_hypotheses = []
                hypotheses_indices = np.searchsorted(hypotheses_partition, indices, side='right')
                for idx, hypothesis_idx in enumerate(hypotheses_indices):
                    best_idx = indices[idx]
                    new_score = hypotheses_scores[best_idx]
                    base_hypothesis = beam.hypotheses[hypothesis_idx]

                    new_sequence = base_hypothesis.sequence
                    if hypothesis_set_lengths[hypothesis_idx] > 1:
                        base_idx = hypotheses_partition[hypothesis_idx - 1] if hypothesis_idx else 0

                        # Make sure to generate a copy of the sequence when adding an element
                        new_sequence = new_sequence + [int(best_idx) - int(base_idx)]
                    new_hypotheses.append(BeamHypothesis(new_sequence, new_score))

                beam.hypotheses = new_hypotheses

    def decode(self, encoded, hidden, beams):
        ''' Decodes the given inputs '''
        self.model.eval()
        with torch.no_grad():
            encoded_hidden_list = utils.split_or_chunk((encoded, hidden.transpose(0, 1)), len(beams))
            while not self.all_done(beams):
                encoded_batch, batch, beam_map = self.collate(encoded_hidden_list, beams)

                logits = []
                chunks = [(encoded_batch, batch)]
                while chunks:
                    try:
                        encoded_batch, batch = chunks.pop()
                        # full_logits = self.model(encoded_batch, batch)

                        full_logits, decoder_hidden, decoder_attn = self.model(batch, encoded_batch[1].transpose(0, 1), encoded_batch[0])

                        # We only care about the logits for the most recently computed token, since
                        # we keep track of the total log probability of each hypothesis in a beam.
                        logits.append(full_logits.transpose(1, 2))
                    except RuntimeError as rte:
                        if 'out of memory' in str(rte):
                            # This is the EAFP (easier to ask forgiveness than permission) approach
                            # to decoding. When the sequences being decoded become too long, it is
                            # possible to start running out of memory trying to decode all the
                            # sequences at once. This may for example happen early in training
                            # before the model has converged to output <EOS> tokens. Just split the
                            # current batch into two chunks and try again.
                            chunks.extend(zip(
                                utils.split_or_chunk(encoded_batch, 2),
                                utils.split_or_chunk(batch, 2)
                            ))

                            # Additionally clear the cache in case the issue is related to allocator
                            # fragmentation.
                            torch.cuda.empty_cache()
                        else:
                            raise rte

                log_prob = torch.cat(logits)
                self.update_beams(log_prob, beam_map)

            return beams
