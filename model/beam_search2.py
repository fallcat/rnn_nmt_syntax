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
    def __init__(self, start_sequence, hidden, initial_score=0., max_length=0, width=4):
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
            sequences.append(hypothesis.sequence.unsqueeze(0))
            # print("sequence", hypothesis.sequence)
            # print("sequence type", type(hypothesis.sequence))
            scores.append(hypothesis.score)
            hiddens.append(hypothesis.hidden)
        # print("lists")
        # print("sequences", sequences)
        # print("scores", scores)
        # print("hiddens", hiddens)

        return torch.cat(sequences), torch.FloatTensor(scores).to(DEVICE), torch.cat(hiddens)


class BeamSearchDecoder(object):
    def __init__(self, decoder, config, initial_score=0.):
        self.decoder = decoder
        self.config = config
        self.initial_score = initial_score

    def initialize_search(self, start_sequences, max_lengths=0, initial_scores=0., beam_width=4):
        ''' Initialize a batch of beams '''
        beams = []
        if isinstance(max_lengths, int):
            max_lengths = [max_lengths] * len(start_sequences)

        if isinstance(initial_scores, int):
            initial_scores = [initial_scores] * len(start_sequences)

        for sequence, score, max_length in zip(start_sequences, initial_scores, max_lengths):
            beams.append(Beam(sequence, score, max_length, beam_width))

        return beams

    def search_all(self, sequences, topv, topi, scores, hiddens):
        new_scores = scores

        # Project each position's beam_width number of candidates to a 2d vector in a beam_width + 2 d space,
        # and broadcast to add together to get the score of each combination.
        for s in range(self.config['span_size']):
            new_scores += topv[:, s].view([len(scores)] + [1]*s + [-1] + [1]*(self.config['beam_width'] - s))
        new_topv, new_topi = new_scores.view(-1).topk(self.config['beam_width'])
        top_indices = [[] for _ in range(len(new_topv))]
        for s in range(self.config['span_size']):
            dims_elements = self.config['beam_width'] ** (self.config['span_size'] - s)
            dim_idx = new_topi // dims_elements
            torch.remainder_(new_topi, dims_elements)
            for i, new_subseq in enumerate(top_indices):
                new_subseq.append(dim_idx[i])
        return [BeamHypothesis(torch.cat(sequences[new_subseq[0]],
                                         topi[new_subseq[0]][range(self.config['span_size']), new_subseq[1:]]),
                               new_topv[i], hiddens[new_subseq[0]])
                for i, new_subseq in enumerate(top_indices)]

    def search_sequential(self, sequences, topv, topi, scores, hiddens):
        for s in range(self.config['span_size']):
            if s == 0:
                print("scores", scores.view(-1, 1).dtype)
                print("topv", topv[:, s, :].dtype)
                newscores = scores.view(-1, 1) + topv[:, s, :]
            else:
                newscores = torch.cat([nc[2] + topv[nc[0], s, :] for nc in new_candidates])
            topsv, topsi = newscores.view(-1).topk(self.config['beam_width'])
            rows, cols = topv[:, s, :].size()
            rowsi = topsi // cols  # indices of the topk beams
            colsi = torch.remainder(topsi, cols)
            if s == 0:
                # each candiate has a tuple of (idx of previously decoded sequence, sequence including this new word,
                # the new word, corresponding hidden layer)
                print("rowsi[i]", rowsi[0])
                print("sequences[rowsi[i]]", sequences[rowsi[0].item()])
                print("topi[rowsi[0], colsi[0]]", topi[rowsi[0], colsi[0]])
                print("colsi[0]", colsi[0])
                print("torch.LongTensor([topi[rowsi[0], colsi[0]]]).to(DEVICE)", torch.LongTensor([topi[rowsi[0].item(), colsi[0].item()]]).to(DEVICE))
                print("torch.cat((sequences[rowsi[i]], torch.LongTensor([topi[rowsi[i], colsi[i]]]).to(DEVICE)))",
                      torch.cat((sequences[rowsi[0].item()], torch.LongTensor([topi[rowsi[0].item(), colsi[0].item()]]).to(DEVICE))))
                print("topsv[i]", topsv[0])
                print("hiddens[rowsi[i]]", hiddens[rowsi[0].item()])
                new_candidates = [(rowsi[i], torch.cat((sequences[rowsi[i].item()], torch.LongTensor([topi[rowsi[i].item(), colsi[i].item()]]).to(DEVICE))),
                                   topsv[i], hiddens[rowsi[i].item()]) for i in range(self.config['beam_width'])]
            else:
                new_candidates = [(new_candidates[rowsi[i]][0],
                                   torch.cat((new_candidates[rowsi[i].item()][1] + torch.LongTensor([topi[rowsi[i].item(), colsi[i].item()]]).to(DEVICE))),
                                   topsv[i],
                                   new_candidates[rowsi[i].item()][3]) for i in range(self.config['beam_width'])]
        return [BeamHypothesis(candidate[1], candidate[2], candidate[3]) for candidate in new_candidates]

    def decode(self, encoder_outputs, encoder_hidden, start_sequences):
        self.decoder.eval()
        with torch.no_grad():
            # print("encoder_outputs", encoder_outputs.size())
            # print("encoder_hidden", encoder_hidden.transpose(0, 1).size())
            encoded_hidden_list = utils.split_or_chunk((encoder_outputs, encoder_hidden.transpose(0, 1)),
                                                       len(encoder_outputs))
            beams = []
            for i, row in enumerate(encoded_hidden_list):
                beam = Beam(start_sequences[i], row[1].transpose(0, 1), self.initial_score,
                            self.config['max_length'], self.config['beam_width'])
                for l in range(int(self.config['max_length']/self.config['span_size'])):
                    sequences, scores, hiddens = beam.collate()
                    # print("sequences", sequences.size())
                    # print("scores", scores.size())
                    # print("hiddens", hiddens.size())
                    decoder_output, decoder_hidden, decoder_attn = self.decoder(sequences[:, -self.config['span_size']:],
                                                                                hiddens.transpose(0, 1),
                                                                                row[0])
                    topv, topi = decoder_output.topk(self.config['beam_width'], dim=2)
                    if self.config['beam_search_all']:
                        new_hypotheses = self.search_all(sequences, topv, topi, scores, decoder_hidden)
                    else:
                        new_hypotheses = self.search_sequential(sequences, topv, topi, scores, decoder_hidden)
                    beam.hypotheses = new_hypotheses
                beams.append(beam)

            return beams

