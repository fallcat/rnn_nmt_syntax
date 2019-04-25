import torch
import torch.nn as nn
import torch.nn.functional as F
from model import PAD_token, SOS_token, EOS_token, MAX_LENGTH, SPAN_SIZE, DEVICE


class BatchEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1, rnn_type="GRU"):
        super(BatchEncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn_type = rnn_type
        if rnn_type == "GRU":
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=self.dropout, batch_first=True)
        else:
            self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=self.dropout, batch_first=True)

    def forward(self, input_seqs, input_lengths, total_length, hidden=None):
        # print("inp", input_seqs.size())
        batch_size = input_seqs.size()[0]
        # print("inside batch encoder")
        # print("inp", input_seqs.size())
        # print("input_seq len", input_seqs.size()[1])
        # print("batch size", batch_size)

        ##### input_seq.new_zeros v
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=DEVICE)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=DEVICE)
        # hidden = input_seqs.new_zeros(self.num_layers, batch_size, self.hidden_size)
        embedded = self.embedding(input_seqs)
        # print("embedded", embedded)
        # print("emb size", embedded.size())
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        # try:
        #     print("packed", packed.data.size())
        # except:
        #     print("packed", packed)
        # print("hidden", hidden.size())
        if self.rnn_type == "GRU":
            self.gru.flatten_parameters()
            output, hidden = self.gru(packed, hidden)
        else:
            self.lstm.flatten_parameters()
            output, (hidden, cell) = self.lstm(packed, (hidden, cell))
        # print("output", output.data.size())
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=total_length)  # unpack (back to padded)
        return output, hidden, cell


class BatchDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=4, dropout_p=0.1, max_length=MAX_LENGTH, span_size=SPAN_SIZE, rnn_type="GRU"):
        super(BatchDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.span_size = 1

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # self.cat_embeddings = nn.Linear(self.hidden_size * self.span_size, self.hidden_size)
        # self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.v = nn.Linear(self.hidden_size, 1)
        # self.attn = nn.Parameter(torch.Tensor(self.hidden_size * 2, self.hidden_size))
        # self.v = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        # gain = nn.init.calculate_gain('linear')
        # nn.init.xavier_uniform_(self.attn, gain)
        # nn.init.xavier_uniform_(self.v, gain)
        # # self.attn = nn.Linear(self.hidden_size * 2, 1)
        # self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn_type = rnn_type
        if rnn_type == "GRU":
            self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout_p,
                              batch_first=True)
        else:
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout_p,
                                batch_first=True)
        # self.grus = nn.ModuleList([nn.GRU(self.hidden_size, self.hidden_size) for _ in range(num_layers)])
        self.out = nn.Linear(self.hidden_size, self.output_size * span_size)

    def forward(self, inputs, hidden, cell, encoder_outputs):
        # Assume inputs is padded to max length, max_length is multiple of span_size
        # ==========================================================================

        bsz = inputs.size()[0]
        encoder_seq_len = encoder_outputs.size()[1]
        # print("bsz", bsz)
        # print("encoder_seq_len", encoder_seq_len)
        embeddeds = self.embedding(inputs)  # B x S -> B x S x H
        # embeddeds = embeddeds.view(bsz, -1)  # B x (S x H)
        embeddeds = self.dropout(embeddeds)  # B x (S x H)
        print("embeddes", embeddeds.size())

        # embeddeds = self.cat_embeddings(embeddeds).unsqueeze(1)

        if self.rnn_type == "GRU":
            self.gru.flatten_parameters()
            rnn_output, hidden = self.gru(embeddeds, hidden)
        else:
            self.lstm.flatten_parameters()
            # print("embeddeds", embeddeds.size())
            # print("hidden", hidden.size())
            # print("cell", cell.size())
            rnn_output, (hidden, cell) = self.lstm(embeddeds, (hidden, cell))

        output = F.relu(rnn_output)
        output = self.out(output)
        output = F.log_softmax(output, dim=2)

        attn_weight = 0

        return output, hidden, cell, attn_weight


class BatchKspanDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=4, dropout_p=0.1, max_length=MAX_LENGTH, span_size=SPAN_SIZE, rnn_type="GRU"):
        super(BatchKspanDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.span_size = span_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.cat_embeddings = nn.Linear(self.hidden_size * self.span_size, self.hidden_size)
        # self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.v = nn.Linear(self.hidden_size, 1)
        # self.attn = nn.Parameter(torch.Tensor(self.hidden_size * 2, self.hidden_size))
        # self.v = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        # gain = nn.init.calculate_gain('linear')
        # nn.init.xavier_uniform_(self.attn, gain)
        # nn.init.xavier_uniform_(self.v, gain)
        # # self.attn = nn.Linear(self.hidden_size * 2, 1)
        # self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn_type = rnn_type
        if rnn_type == "GRU":
            self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout_p,
                              batch_first=True)
        else:
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout_p,
                                batch_first=True)
        # self.grus = nn.ModuleList([nn.GRU(self.hidden_size, self.hidden_size) for _ in range(num_layers)])
        self.out = nn.Linear(self.hidden_size, self.output_size * span_size)

    def forward(self, inputs, hidden, cell, encoder_outputs):
        # Assume inputs is padded to max length, max_length is multiple of span_size
        # ==========================================================================

        bsz = inputs.size()[0]
        encoder_seq_len = encoder_outputs.size()[1]
        # print("bsz", bsz)
        # print("encoder_seq_len", encoder_seq_len)
        embeddeds = self.embedding(inputs)  # B x S -> B x S x H
        embeddeds = embeddeds.view(bsz, -1)  # B x (S x H)
        embeddeds = self.dropout(embeddeds)  # B x (S x H)

        embeddeds = self.cat_embeddings(embeddeds).unsqueeze(1)

        if self.rnn_type == "GRU":
            self.gru.flatten_parameters()
            rnn_output, hidden = self.gru(embeddeds, hidden)
        else:
            self.lstm.flatten_parameters()
            # print("embeddeds", embeddeds.size())
            # print("hidden", hidden.size())
            # print("cell", cell.size())
            rnn_output, (hidden, cell) = self.lstm(embeddeds, (hidden, cell))

        # concatted = torch.cat((
        #     rnn_output.expand((rnn_output.size()[0], encoder_seq_len, rnn_output.size()[2])),
        #     encoder_outputs), 2)
        #
        # concatted_size = concatted.size()
        #
        # attn_weight = F.softmax(torch.chain_matmul(concatted.view(-1, concatted.size()[2]), self.attn, self.v).view(concatted_size[0], concatted_size[1], -1), dim=1)
        #
        # attn_applied = torch.bmm(attn_weight.transpose(1, 2), encoder_outputs) # B x 1 x H
        #
        # output = torch.cat((rnn_output, attn_applied), 2)
        #
        # output = self.attn_combine(output)

        output = F.relu(rnn_output)
        output = self.out(output).view(bsz, self.span_size, -1)
        output = F.log_softmax(output, dim=2)

        attn_weight = 0

        return output, hidden, cell, attn_weight


class BatchAttnKspanDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=4, dropout_p=0.1, max_length=MAX_LENGTH, span_size=SPAN_SIZE, rnn_type="GRU"):
        super(BatchAttnKspanDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.span_size = span_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.cat_embeddings = nn.Linear(self.hidden_size * self.span_size, self.hidden_size)
        # self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.v = nn.Linear(self.hidden_size, 1)
        self.attn = nn.Parameter(torch.Tensor(self.hidden_size * 2, self.hidden_size))
        self.v = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_uniform_(self.attn, gain)
        nn.init.xavier_uniform_(self.v, gain)
        # self.attn = nn.Linear(self.hidden_size * 2, 1)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn_type = rnn_type
        if rnn_type == "GRU":
            self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout_p, batch_first=True)
        else:
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout_p,
                                batch_first=True)
        # self.grus = nn.ModuleList([nn.GRU(self.hidden_size, self.hidden_size) for _ in range(num_layers)])
        self.out = nn.Linear(self.hidden_size, self.output_size * span_size)

    def forward(self, inputs, hidden, cell, encoder_outputs):
        # Assume inputs is padded to max length, max_length is multiple of span_size
        # ==========================================================================

        bsz = inputs.size()[0]
        encoder_seq_len = encoder_outputs.size()[1]
        # print("bsz", bsz)
        # print("encoder_seq_len", encoder_seq_len)
        embeddeds = self.embedding(inputs)  # B x S -> B x S x H
        embeddeds = embeddeds.view(bsz, -1)  # B x (S x H)
        embeddeds = self.dropout(embeddeds)  # B x (S x H)

        embeddeds = self.cat_embeddings(embeddeds).unsqueeze(1)

        if self.rnn_type == "GRU":
            self.gru.flatten_parameters()
            rnn_output, hidden = self.gru(embeddeds, hidden)
        else:
            self.lstm.flatten_parameters()
            # print("embeddeds", embeddeds.size())
            # print("hidden", hidden.size())
            # print("cell", cell.size())
            rnn_output, (hidden, cell) = self.lstm(embeddeds, (hidden, cell))

        concatted = torch.cat((
            rnn_output.expand((rnn_output.size()[0], encoder_seq_len, rnn_output.size()[2])),
            encoder_outputs), 2)

        concatted_size = concatted.size()

        attn_weight = F.softmax(torch.chain_matmul(concatted.view(-1, concatted.size()[2]), self.attn, self.v).view(concatted_size[0], concatted_size[1], -1), dim=1)

        attn_applied = torch.bmm(attn_weight.transpose(1, 2), encoder_outputs) # B x 1 x H

        output = torch.cat((rnn_output, attn_applied), 2)

        output = self.attn_combine(output)

        output = F.relu(output)
        output = self.out(output).view(bsz, self.span_size, -1)
        output = F.log_softmax(output, dim=2)

        return output, hidden, cell, attn_weight

class BatchAttnKspanDecoderRNN3(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=4, dropout_p=0.1, max_length=MAX_LENGTH, span_size=SPAN_SIZE):
        super(BatchAttnKspanDecoderRNN3, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.span_size = span_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.cat_embeddings = nn.Linear(self.hidden_size * self.span_size, self.hidden_size)
        # self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.v = nn.Linear(self.hidden_size, 1)
        self.attn = nn.Parameter(torch.Tensor(self.hidden_size * 2, self.hidden_size))
        self.v = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_uniform_(self.attn, gain)
        nn.init.xavier_uniform_(self.v, gain)
        # self.attn = nn.Linear(self.hidden_size * 2, 1)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout_p, batch_first=True)
        # self.grus = nn.ModuleList([nn.GRU(self.hidden_size, self.hidden_size) for _ in range(num_layers)])
        self.out = nn.Linear(self.hidden_size, self.output_size * span_size)

    def forward(self, inputs, hidden, encoder_outputs):
        # Assume inputs is padded to max length, max_length is multiple of span_size
        # ==========================================================================

        bsz = inputs.size()[0]
        encoder_seq_len = encoder_outputs.size()[1]
        # print("bsz", bsz)
        # print("encoder_seq_len", encoder_seq_len)
        embeddeds = self.embedding(inputs)  # B x S -> B x S x H
        embeddeds = embeddeds.view(bsz, -1)  # B x (S x H)
        embeddeds = self.dropout(embeddeds)  # B x (S x H)

        embeddeds = self.cat_embeddings(embeddeds).unsqueeze(1)

        self.gru.flatten_parameters()
        rnn_output, hidden = self.gru(embeddeds, hidden)

        concatted = torch.cat((
            rnn_output.expand((rnn_output.size()[0], encoder_seq_len, rnn_output.size()[2])),
            encoder_outputs), 2)

        concatted_size = concatted.size()

        attn_weight = F.softmax(torch.chain_matmul(concatted.view(-1, concatted.size()[2]), self.attn, self.v).view(concatted_size[0], concatted_size[1], -1), dim=1)

        attn_applied = torch.bmm(attn_weight.transpose(1, 2), encoder_outputs) # B x 1 x H

        output = torch.cat((rnn_output, attn_applied), 2)

        output = self.attn_combine(output)

        output = F.relu(output)
        output = self.out(output).view(bsz, self.span_size, -1)
        output = F.log_softmax(output, dim=2)

        return output, hidden, attn_weight

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
#         print('[attn] seq len', seq_len)
#         print('[attn] encoder_outputs', encoder_outputs.size()) # S x B x N
#         print('[attn] hidden', hidden.size()) # S=1 x B x N

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(batch_size, seq_len)) # B x S
#         print('[attn] attn_energies', attn_energies.size())

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(batch_size):
            # Calculate energy for each encoder output
            for i in range(seq_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
#         print('[attn] attn_energies', attn_energies.size())
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time (in order to do teacher forcing)

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
#         print('[decoder] input_seq', input_seq.size()) # batch_size x 1
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N
#         print('[decoder] word_embedded', embedded.size())

        # Get current hidden state from input word and last hidden state
#         print('[decoder] last_hidden', last_hidden.size())
        rnn_output, hidden = self.gru(embedded, last_hidden)
#         print('[decoder] rnn_output', rnn_output.size())

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
#         print('[decoder] attn_weights', attn_weights.size())
#         print('[decoder] encoder_outputs', encoder_outputs.size())
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N
#         print('[decoder] context', context.size())

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
#         print('[decoder] rnn_output', rnn_output.size())
#         print('[decoder] context', context.size())
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6)
#         output = F.log_softmax(self.out(concat_output))
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights





class BatchBahdanauAttnKspanDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=4, dropout_p=0.1, max_length=MAX_LENGTH, span_size=SPAN_SIZE):
        super(BatchBahdanauAttnKspanDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.span_size = span_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.cat_embeddings = nn.Linear(self.hidden_size * self.span_size, self.hidden_size)
        # self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.v = nn.Linear(self.hidden_size, 1)
        self.attn = nn.Parameter(torch.Tensor(self.hidden_size * 2, self.hidden_size))
        self.v = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_uniform_(self.attn, gain)
        nn.init.xavier_uniform_(self.v, gain)
        # self.attn = nn.Linear(self.hidden_size * 2, 1)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout_p, batch_first=True)
        # self.grus = nn.ModuleList([nn.GRU(self.hidden_size, self.hidden_size) for _ in range(num_layers)])
        self.out = nn.Linear(self.hidden_size, self.output_size * span_size)

    def forward(self, inputs, hidden, cell, encoder_outputs):
        # Assume inputs is padded to max length, max_length is multiple of span_size
        # ==========================================================================

        bsz = inputs.size()[0]
        encoder_seq_len = encoder_outputs.size()[1]
        # print("bsz", bsz)
        # print("encoder_seq_len", encoder_seq_len)
        embeddeds = self.embedding(inputs)  # B x S -> B x S x H
        embeddeds = embeddeds.view(bsz, -1)  # B x (S x H)
        embeddeds = self.dropout(embeddeds)  # B x (S x H)

        embeddeds = self.cat_embeddings(embeddeds).unsqueeze(1)
        # print("embeddeds", embeddeds.size())


        hidden_size = hidden.size()
        concatted = torch.cat((
            hidden[-1].unsqueeze(1).expand((hidden_size[1], encoder_seq_len, hidden_size[2])),
            encoder_outputs), 2)

        concatted_size = concatted.size()

        attn_weight = F.softmax(torch.chain_matmul(concatted.view(-1, concatted.size()[2]), self.attn, self.v).view(concatted_size[0], concatted_size[1], -1), dim=1)

        attn_applied = torch.bmm(attn_weight.transpose(1, 2), encoder_outputs) # B x 1 x H

        embeddeds = torch.cat((embeddeds, attn_applied), 2)

        embeddeds = self.attn_combine(embeddeds)

        self.gru.flatten_parameters()
        rnn_output, hidden = self.gru(embeddeds, hidden)

        output = F.relu(rnn_output)
        output = self.out(output).view(bsz, self.span_size, -1)
        output = F.log_softmax(output, dim=2)

        return output, hidden, cell, attn_weight