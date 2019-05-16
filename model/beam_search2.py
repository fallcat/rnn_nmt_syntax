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

        return torch.cat(sequences, 0), torch.tensor(scores, dtype=torch.float32).to(DEVICE), \
               (torch.cat(hiddens, 0), torch.cat(cells, 0))


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

    def collate(self, encoder_outputs, beams):
        sequences = []
        scores = []
        hiddens = []
        cells = []
        encoder_batch = []
        for i, beam in enumerate(beams):
            sequence, score, hidden = beam.collate()
            sequences.append(sequence)
            scores.append(score)
            hiddens.append(hidden[0])
            cells.append(hidden[1])
            encoder_batch.append(encoder_outputs[i].unsqueeze(0).expand(sequence.size()[0],
                                                                        encoder_outputs[i].size()[0],
                                                                        encoder_outputs[i].size()[1]))
        return torch.cat(sequences, 0), torch.cat(scores, 0), (torch.cat(hiddens, 0), torch.cat(cells, 0)), \
               torch.cat(encoder_batch, 0)

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
                newscores = scores.view(-1, 1) + topv[:, s, :]
            else:
                newscores = torch.cat([nc[2] + topv[nc[0], s, :] for nc in new_candidates])
            topsv, topsi = newscores.view(-1).topk(self.config['beam_width'])
            rowsi = topsi // self.config['beam_width']  # indices of the topk beams
            colsi = topsi.remainder(self.config['beam_width'])
            if s == 0:
                # print("rowsi", rowsi)
                # print("colsi", colsi)
                # print("topi", topi.size())
                new_candidates = [(rowsi[i],
                                   torch.cat((sequences[rowsi[i]], topi[rowsi[i], s, colsi[i]].to('cpu').unsqueeze(0))),
                                   topsv[i],
                                   (hiddens[0][:, rowsi[i]], hiddens[1][:, rowsi[i]]))
                                  for i in range(self.config['beam_width'])]
                new_candidates = [(nc[0],
                                   nc[1],
                                   self.normalized_score(nc[2],
                                                         nc[1][:nc[1].numpy().tolist().index(EOS_token)].size()[0]),
                                   nc[3]) if EOS_token in nc[1] else nc for nc in new_candidates]
            else:
                new_candidates = [(new_candidates[rowsi[i]][0],
                                   torch.cat((new_candidates[rowsi[i]][1],
                                              topi[new_candidates[rowsi[i]][0], s, colsi[i]].to('cpu').unsqueeze(0))),
                                   topsv[i],
                                   new_candidates[rowsi[i]][3]) for i in range(self.config['beam_width'])]
                new_candidates = [(nc[0],
                                   nc[1],
                                   self.normalized_score(nc[2],
                                                         nc[1][:nc[1].numpy().tolist().index(EOS_token)].size()[0]),
                                   nc[3]) if EOS_token in nc[1] else nc for nc in new_candidates]
        return [BeamHypothesis(candidate[1], candidate[2], candidate[3]) for candidate in new_candidates]

    def search_sequential_batch(self, sequences, topv, topi, scores, hiddens, batch_size):
        print("utils.split_or_chunk((sequences, topv, topi, scores,hiddens[0], hiddens[1]),batch_size)", len(utils.split_or_chunk((sequences, topv, topi, scores,hiddens[0], hiddens[1]),batch_size)))
        for item in utils.split_or_chunk((sequences, topv, topi, scores, hiddens[0], hiddens[1]), batch_size):
            print("size", item.size())
        sequences_l, topv_l, topi_l, scores_l, hiddens_l, cells_l = utils.split_or_chunk((sequences, topv, topi, scores,
                                                                                          hiddens[0], hiddens[1]),
                                                                                          batch_size)
        for b in range(batch_size):
            for s in range(self.config['span_size']):
                if s == 0:
                    newscores = scores_l[b].view(-1, 1) + topv_l[b][:, s, :]
                else:
                    newscores = torch.cat([nc[2] + topv_l[b][nc[0], s, :] for nc in new_candidates])
                topsv, topsi = newscores.view(-1).topk(self.config['beam_width'])
                rowsi = topsi // self.config['beam_width']  # indices of the topk beams
                colsi = topsi.remainder(self.config['beam_width'])
                if s == 0:
                    # print("rowsi", rowsi)
                    # print("colsi", colsi)
                    # print("topi", topi.size())
                    new_candidates = [(rowsi[i],
                                       torch.cat((sequences_l[b][rowsi[i]], topi_l[b][rowsi[i], s, colsi[i]].to('cpu').unsqueeze(0))),
                                       topsv[i],
                                       (hiddens_l[b][:, rowsi[i]], cells_l[b][:, rowsi[i]]))
                                      for i in range(self.config['beam_width'])]
                    new_candidates = [(nc[0],
                                       nc[1],
                                       self.normalized_score(nc[2],
                                                             nc[1][:nc[1].numpy().tolist().index(EOS_token)].size()[0]),
                                       nc[3]) if EOS_token in nc[1] else nc for nc in new_candidates]
                else:
                    new_candidates = [(new_candidates[rowsi[i]][0],
                                       torch.cat((new_candidates[rowsi[i]][1],
                                                  topi_l[b][new_candidates[rowsi[i]][0], s, colsi[i]].to('cpu').unsqueeze(0))),
                                       topsv[i],
                                       new_candidates[rowsi[i]][3]) for i in range(self.config['beam_width'])]
                    new_candidates = [(nc[0],
                                       nc[1],
                                       self.normalized_score(nc[2],
                                                             nc[1][:nc[1].numpy().tolist().index(EOS_token)].size()[0]),
                                       nc[3]) if EOS_token in nc[1] else nc for nc in new_candidates]
            yield [BeamHypothesis(candidate[1], candidate[2], candidate[3]) for candidate in new_candidates]

    def decode_batch(self, encoder_outputs, encoder_hidden, start_sequences):
        self.decoder.eval()
        batch_size = len(encoder_outputs)
        with torch.no_grad():
            decoder_hidden = torch.zeros(self.config['num_layers'] + 1 + self.config['more_decoder_layers'],
                                         batch_size, self.config['hidden_size'],
                                         device=DEVICE)
            decoder_cell = torch.zeros(self.config['num_layers'] + 1 + self.config['more_decoder_layers'],
                                       batch_size, self.config['hidden_size'],
                                       device=DEVICE)
            encoded_hidden_list = utils.split_or_chunk((encoder_outputs, decoder_hidden.transpose(0, 1),
                                                        decoder_cell.transpose(0, 1)),
                                                       batch_size)
            beams = [Beam(start_sequences[i], (row[1].transpose(0, 1), row[2].transpose(0, 1)), self.initial_score,
                            self.config['max_length'], self.config['beam_width']) for i, row in enumerate(encoded_hidden_list)]

            for l in range(int(self.config['max_length']/self.config['span_size'])):
                sequences, scores, hiddens, encoder_batch = self.collate(encoder_outputs, beams)
                len_seq = sequences.size()[0]
                decoder_output, decoder_hidden, decoder_cell, decoder_attn \
                    = self.decoder(sequences[:, -self.config['span_size']:],
                                   hiddens[0].view(
                                       len_seq,
                                       self.config['num_layers'] + 1 + self.config['more_decoder_layers'],
                                       -1).transpose(0, 1),
                                   hiddens[1].view(
                                       len_seq,
                                       self.config['num_layers'] + 1 + self.config['more_decoder_layers'],
                                       -1).transpose(0, 1),
                                   encoder_batch)
                topv, topi = decoder_output.topk(self.config['beam_width'], dim=2)
                if self.config['beam_search_all']:
                    new_hypotheses = self.search_all(sequences, topv, topi, scores, (decoder_hidden, decoder_cell))
                else:
                    new_hypotheses = self.search_sequential_batch(sequences, topv, topi, scores,
                                                                  (decoder_hidden, decoder_cell), batch_size)
                for i, new_hypothesis in enumerate(new_hypotheses):
                    beams[i].hypotheses = new_hypothesis

            return beams

    def decode(self, encoder_outputs, encoder_hidden, start_sequences):
        self.decoder.eval()
        with torch.no_grad():
            decoder_hidden = torch.zeros(self.config['num_layers'] + 1 + self.config['more_decoder_layers'],
                                         len(encoder_outputs), self.config['hidden_size'],
                                         device=DEVICE)
            decoder_cell = torch.zeros(self.config['num_layers'] + 1 + self.config['more_decoder_layers'],
                                       len(encoder_outputs), self.config['hidden_size'],
                                       device=DEVICE)
            encoded_hidden_list = utils.split_or_chunk((encoder_outputs, decoder_hidden.transpose(0, 1),
                                                        decoder_cell.transpose(0, 1)),
                                                       len(encoder_outputs))
            beams = []
            for i, row in enumerate(encoded_hidden_list):
                beam = Beam(start_sequences[i], (row[1].transpose(0, 1), row[2].transpose(0, 1)), self.initial_score,
                            self.config['max_length'], self.config['beam_width'])
                for l in range(int(self.config['max_length']/self.config['span_size'])):
                    sequences, scores, hiddens = beam.collate()
                    len_seq = sequences.size()[0]
                    decoder_output, decoder_hidden, decoder_cell, decoder_attn \
                        = self.decoder(sequences[:, -self.config['span_size']:],
                                       hiddens[0].view(
                                           len_seq,
                                           self.config['num_layers'] + 1 + self.config['more_decoder_layers'],
                                           -1).transpose(0, 1),
                                       hiddens[1].view(
                                           len_seq,
                                           self.config['num_layers'] + 1 + self.config['more_decoder_layers'],
                                           -1).transpose(0, 1),
                                       row[0].expand(sequences.size()[0], row[0].size()[1], row[0].size()[2]))
                    topv, topi = decoder_output.topk(self.config['beam_width'], dim=2)
                    if self.config['beam_search_all']:
                        new_hypotheses = self.search_all(sequences, topv, topi, scores, (decoder_hidden, decoder_cell))
                    else:
                        new_hypotheses = self.search_sequential(sequences, topv, topi, scores, (decoder_hidden, decoder_cell))
                    beam.hypotheses = new_hypotheses
                beams.append(beam)

            return beams

