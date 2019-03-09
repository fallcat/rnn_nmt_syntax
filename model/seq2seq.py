import torch
import torch.nn as nn
import torch.nn.functional as F
from rnn_nmt_syntax.model import PAD_token, SOS_token, EOS_token, MAX_LENGTH, SPAN_SIZE, DEVICE


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers)
        # self.grus = nn.ModuleList([nn.GRU(self.hidden_size, self.hidden_size) for _ in range(num_layers)])

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        # for i, gru in enumerate(self.grus):
        #     output, hiddens[i] = gru(output, hiddens[i])
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=DEVICE)
        # return [torch.zeros(1, 1, self.hidden_size, device=device) for _ in range(self.num_layers)]


class BatchEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(BatchEncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=self.dropout, batch_first=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # print("inp", input_seqs.size())
        batch_size = input_seqs.size()[0]
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=DEVICE)
        embedded = self.embedding(input_seqs)
        # print("embedded", embedded)
        # print("emb size", embedded.size())
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        # print("packed", packed.data.size())
        # print("hidden", hidden.size())
        output, hidden = self.gru(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # unpack (back to padded)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


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


class AttnKspanDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=4, dropout_p=0.1, max_length=MAX_LENGTH, span_size=SPAN_SIZE):
        super(AttnKspanDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.span_size = span_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * (1 + span_size), self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * (1 + span_size), self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers)
        # self.grus = nn.ModuleList([nn.GRU(self.hidden_size, self.hidden_size) for _ in range(num_layers)])
        self.out = nn.Linear(self.hidden_size, self.output_size * span_size)

    def forward(self, input, hidden, encoder_outputs):
        embeddeds = tuple([self.embedding(input_i).view(1, 1, -1)[0] for input_i in input])
        embeddeds = torch.cat(embeddeds, 1)
        embeddeds = self.dropout(embeddeds)
        attn_weights = torch.cat((embeddeds, hidden[0]), 1)
        attn_weights = self.attn(attn_weights)
        attn_weights = F.softmax(attn_weights, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embeddeds, attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        # for i, gru in enumerate(self.grus):
        #     output, hiddens[i] = gru(output, hiddens[i])
        output = self.out(output[0])
        output = torch.split(output, self.output_size, dim=1)

        output = tuple([F.log_softmax(output_i, dim=1) for output_i in output])
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=DEVICE)


class BatchAttnKspanDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=4, dropout_p=0.1, max_length=MAX_LENGTH, span_size=SPAN_SIZE):
        super(BatchAttnKspanDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.span_size = span_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * (1 + self.span_size), self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * (1 + self.span_size), self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)
        # self.grus = nn.ModuleList([nn.GRU(self.hidden_size, self.hidden_size) for _ in range(num_layers)])
        self.out = nn.Linear(self.hidden_size, self.output_size * span_size)

    def forward(self, inputs, hidden, encoder_outputs):
        # Assume inputs is padded to max length, max_length is multiple of span_size
        # ==========================================================================

        # inputs = [(len(inp) % self.span_size + self.span_size) for inp in inputs]
        bsz, seq_len = inputs.size()
        encoder_seq_len = encoder_outputs.size()[1]
        span_seq_len = int(seq_len/self.span_size)
        embeddeds = self.embedding(inputs)
        embeddeds = embeddeds.view(bsz, span_seq_len, -1)
        # embeddeds = tuple([self.embedding(input_i).view(1, 1, -1)[0] for input_i in input])
        # embeddeds = torch.cat(embeddeds, 1)
        embeddeds = self.dropout(embeddeds)
        # hiddens = torch.zeros(span_seq_len, bsz, self.hidden_size, device=DEVICE)
        outputs = torch.zeros(bsz, span_seq_len, self.hidden_size, device=DEVICE)
        attn_weights = torch.zeros(span_seq_len, bsz, self.max_length)

        for l in range(seq_len):
            print("emb", embeddeds[:,l].size())
            print("hidden", hidden.size())
            attn_weight = F.softmax(self.attn(torch.cat((embeddeds[:, l].contiguous(), hidden[-1]), 1)), dim=1)
            print("attn_weight", attn_weight.size())
            print("encoder_outputs", encoder_outputs.size())
            attn_weights[l] = attn_weight
            print("attn_weight.unsqueeze(1)", attn_weight.unsqueeze(1).size())
            print("encoder_outputs", encoder_outputs.size())
            attn_applied = torch.bmm(attn_weight.unsqueeze(1), encoder_outputs)

            print("embeddeds", embeddeds[:, l].size())
            print("attn_applied", attn_applied.size())
            output = torch.cat((embeddeds[:, l].unsqueeze(1), attn_applied), 1)
            print("output", output.size())
            output = self.attn_combine(output)

            output = F.relu(output)
            print("output", output.size())
            print("hidden", hidden.size())
            output, hidden = self.gru(output.transpose(0, 1), hidden)
            outputs[:, l:l + 1] = output
            #             outputs.append(output)
            # hiddens[l] = hidden
        # for i, gru in enumerate(self.grus):
        #     output, hiddens[i] = gru(output, hiddens[i])
        outputs = self.out(outputs).view(bsz, seq_len, -1)
        outputs = F.log_softmax(outputs, dim=2)
        # output = torch.split(output, self.output_size, dim=1)

        # output = tuple([F.log_softmax(output_i, dim=1) for output_i in output])
        return outputs, hidden, attn_weights


class AttnKspanLSTMDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=4, dropout_p=0.1, max_length=MAX_LENGTH, span_size=SPAN_SIZE):
        super(AttnKspanLSTMDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.span_size = span_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * (1 + span_size), self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * (1 + span_size), self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size * span_size)

    def forward(self, input, hidden, encoder_outputs):
        embeddeds = tuple([self.embedding(input_i).view(1, 1, -1)[0] for input_i in input])
        embeddeds = torch.cat(embeddeds, 1)
        embeddeds = self.dropout(embeddeds)
        attn_weights = torch.cat((embeddeds, hidden[0]), 1)
        attn_weights = self.attn(attn_weights)
        attn_weights = F.softmax(attn_weights, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embeddeds, attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output[0])
        output = torch.split(output, self.output_size, dim=1)

        output = tuple([F.log_softmax(output_i, dim=1) for output_i in output])
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=DEVICE)
