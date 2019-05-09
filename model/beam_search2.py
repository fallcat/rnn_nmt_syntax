import torch
from model import utils, DEVICE, EOS_token


class BeamHypothesis(object):
    def __init__(self, sequence, score, hidden):
        self.sequence = sequence
        self.score = score
        self.hidden = hidden
        self.finish = False

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
        cells = []
        for hypothesis in self.hypotheses:
            sequences.append(hypothesis.sequence.unsqueeze(0))
            # print("sequence", hypothesis.sequence)
            # print("sequence type", type(hypothesis.sequence))
            scores.append(hypothesis.score)
            hiddens.append(hypothesis.hidden[0].unsqueeze(0))
            cells.append(hypothesis.hidden[1].unsqueeze(0))
        # print("lists")
        # print("sequences", len(sequences), sequences[0].size())
        # print("scores", scores[0])
        # print("hiddens", len(hiddens), hiddens[0].size())

        return torch.cat(sequences, 0), torch.FloatTensor(scores).to(DEVICE), (torch.cat(hiddens, 0), torch.cat(cells, 0))


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

    def normalized_score(self, score, length):
        """
        Calculate the normalized score of the hypothesis

        https://arxiv.org/abs/1609.08144
        See equation #14
        """
        return score * ((5 + 1) / (5 + length)) ** self.config['length_penalty']

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
                print("scores", scores.size())
                print("topv", topv[:, s, :].size())
                print("scores", scores)
                print("topv", topv[:, s, :])
                newscores = scores.view(-1, 1) + topv[:, s, :]
            else:
                newscores = torch.cat([nc[2] + topv[nc[0], s, :] for nc in new_candidates])
            topsv, topsi = newscores.view(-1).topk(self.config['beam_width'])
            rowsi = topsi // self.config['beam_width']  # indices of the topk beams
            colsi = topsi.remainder(self.config['beam_width'])
            print("rowsi", rowsi)
            print("colsi", colsi)
            print("new scores", newscores)
            if s == 0:
                # print("rowsi", rowsi)
                # print("colsi", colsi)
                # print("topi", topi.size())
                new_candidates = [(rowsi[i],
                                   torch.cat((sequences[rowsi[i]], topi[rowsi[i], s, colsi[i]].to('cpu').unsqueeze(0))),
                                   topsv[i],
                                   (hiddens[0][:, rowsi[i]], hiddens[1][:, rowsi[i]]))
                                  for i in range(self.config['beam_width'])]
                # new_candidates = [(nc[0], nc[1], self.normalized_score(nc[2], len(nc[1][:nc[1].numpy().tolist().index(EOS_token)])),
                #                    nc[3]) if EOS_token in nc[1] else nc for nc in new_candidates]
            else:
                # print("rowsi", rowsi)
                # print("colsi", colsi)
                # print("topi", topi.size())
                # print("new_candidates[rowsi[i]][0]", new_candidates[rowsi[0]][0].size())
                # print("torch.cat((new_candidates[rowsi[i]][1], topi[rowsi[i], colsi[i], topsi[i]].unsqueeze(0)))", torch.cat((new_candidates[rowsi[0]][1], topi[rowsi[0], colsi[0], topsi[0]].unsqueeze(0))).size())
                # print("topsv[i]", topsv[0].size())
                # print("new_candidates[rowsi[i]][3]", new_candidates[rowsi[0]][3].size())
                # for i in range(self.config['beam_width']):
                #     print("new_candidates[rowsi[i]][0]", new_candidates[rowsi[i]][0])
                #     print("new_candidates[rowsi[i]][1]", new_candidates[rowsi[i]][1])
                #     print("topi[rowsi[i], s, colsi[i]]", topi[new_candidates[rowsi[i]][0], s, colsi[i]])
                #     print("topi[rowsi[i], s, colsi[i]].to('cpu').unsqueeze(0)", topi[new_candidates[rowsi[i]][0], s, colsi[i]].to('cpu').unsqueeze(0))
                #     print("torch.cat((new_candidates[rowsi[i]][1], topi[rowsi[i], s, colsi[i]].to('cpu').unsqueeze(0)))", torch.cat((new_candidates[rowsi[i]][1], topi[new_candidates[rowsi[i]][0], s, colsi[i]].to('cpu').unsqueeze(0))))
                #     print("topsv[i]", topsv[i])
                #     print("new_candidates[rowsi[i]][3]", new_candidates[rowsi[i]][3])
                new_candidates = [(new_candidates[rowsi[i]][0],
                                   torch.cat((new_candidates[rowsi[i]][1], topi[new_candidates[rowsi[i]][0], s, colsi[i]].to('cpu').unsqueeze(0))),
                                   topsv[i],
                                   new_candidates[rowsi[i]][3]) for i in range(self.config['beam_width'])]
                # new_candidates = [(nc[0], nc[1], self.normalized_score(nc[2], len(nc[1][:nc[1].numpy().tolist().index(EOS_token)])),
                #                    nc[3]) if EOS_token in nc[1] else nc for nc in new_candidates]
        return [BeamHypothesis(candidate[1], candidate[2], candidate[3]) for candidate in new_candidates]

    def decode(self, encoder_outputs, encoder_hidden, start_sequences, index2word):
        self.decoder.eval()
        with torch.no_grad():
            # print("encoder_outputs", encoder_outputs.size())
            # print("encoder_hidden", encoder_hidden.transpose(0, 1).size())
            decoder_cell = torch.zeros(self.config['num_layers'], len(encoder_outputs), self.config['hidden_size'],
                                       device=DEVICE)
            # print("encoder_hidden", encoder_hidden.size())
            # print("decoder_cell", decoder_cell.size())
            encoded_hidden_list = utils.split_or_chunk((encoder_outputs, encoder_hidden.transpose(0, 1),
                                                        decoder_cell.transpose(0, 1)),
                                                       len(encoder_outputs))
            beams = []
            for i, row in enumerate(encoded_hidden_list):
                print("i", i)
                beam = Beam(start_sequences[i], (row[1].transpose(0, 1), row[2].transpose(0, 1)), self.initial_score,
                            self.config['max_length'], self.config['beam_width'])
                for l in range(int(self.config['max_length']/self.config['span_size'])):
                    print("l", l)
                    sequences, scores, hiddens = beam.collate()
                    print("hiddens", hiddens[0].size())
                    decoder_output, decoder_hidden, decoder_cell, decoder_attn = self.decoder(sequences[:, -self.config['span_size']:],
                                                                                              hiddens[0].view(
                                                                                                  len(sequences),
                                                                                                  self.config[
                                                                                                      'num_layers'],
                                                                                                  -1).transpose(0, 1),
                                                                                              hiddens[1].view(
                                                                                                  len(sequences),
                                                                                                  self.config[
                                                                                                      'num_layers'],
                                                                                                  -1).transpose(0, 1),
                                                                                              row[0].expand(len(sequences),
                                                                                                            row[0].size()[1],
                                                                                                            row[0].size()[2]))
                    topv, topi = decoder_output.topk(self.config['beam_width'], dim=2)
                    print("topv outside", topv)
                    print("topi outside", topi)
                    if self.config['beam_search_all']:
                        new_hypotheses = self.search_all(sequences, topv, topi, scores, (decoder_hidden, decoder_cell))
                    else:
                        new_hypotheses = self.search_sequential(sequences, topv, topi, scores, (decoder_hidden, decoder_cell))
                    beam.hypotheses = new_hypotheses
                    for h in new_hypotheses:
                        print("sequence", [index2word[w.item()] for w in h.sequence], "score", h.score)
                beams.append(beam)

            return beams

