import torch
import torch.nn as nn
import torch.nn.functional as F
from model import PAD_token, SOS_token, EOS_token, MAX_LENGTH, SPAN_SIZE, DEVICE


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_p=0.1, rnn_type="GRU", num_directions=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout_p

        self.embedding = nn.Embedding(input_size, hidden_size)  # no dropout as only one layer!

        self.rnn_type = rnn_type

        if rnn_type == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_seqs, input_lengths, total_length):
        # src = [src sent len, batch size]

        embedded = self.dropout(self.embedding(input_seqs))
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)

        # embedded = [src sent len, batch size, emb dim]
        outputs, hidden = self.rnn(packed)  # no cell state!

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=total_length)

        if self.rnn_type == "GRU":
            cell = torch.zeros(self.num_layers, input_seqs.size()[0], self.hidden_size, device=DEVICE)
        else:
            cell = hidden[1]
            hidden = hidden[0]

        return outputs, hidden, cell


class BatchEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_p=0.1, rnn_type="GRU", num_directions=1):
        super(BatchEncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)
        self.convert = nn.Linear(2, 1)

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn_type = rnn_type
        if rnn_type == "GRU":
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=self.dropout_p, batch_first=True,
                              bidirectional=(num_directions == 2))
            for name, param in self.gru.named_parameters():
                if 'bias' or 'weight' in name:
                    nn.init.uniform_(param, -0.1, 0.1)
        else:
            self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=self.dropout_p, batch_first=True,
                                bidirectional=(num_directions == 2))
            for name, param in self.lstm.named_parameters():
                if 'bias' or 'weight' in name:
                    nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, input_seqs, input_lengths, total_length, hidden=None):
        batch_size = input_seqs.size()[0]

        hidden = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=DEVICE)
        cell = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=DEVICE)
        embedded = self.embedding(input_seqs)
        embedded = self.dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)

        if self.rnn_type == "GRU":
            self.gru.flatten_parameters()
            output, hidden = self.gru(packed, hidden)
        else:
            self.lstm.flatten_parameters()
            output, (hidden, cell) = self.lstm(packed, (hidden, cell))
        # print("output", output.data.size())
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=total_length)  # unpack (back to padded)
        if self.num_directions == 2:
            hidden = self.convert(hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
                                  .transpose(1, 3)).transpose(1,3).squeeze(1)
        return output, hidden, cell


class BatchBahdanauEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_p=0.1, max_length=MAX_LENGTH, rnn_type="GRU", num_directions=1):
        super(BatchBahdanauEncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)
        self.convert = nn.Linear(2, 1)
        self.max_length = max_length

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn_type = rnn_type
        if rnn_type == "GRU":
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=self.dropout_p, batch_first=True,
                              bidirectional=(num_directions == 2))

        else:
            self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=self.dropout_p, batch_first=True,
                                bidirectional=(num_directions == 2))

    def forward(self, input_seqs, input_lengths, total_length, hidden=None):
        batch_size = input_seqs.size()[0]

        hidden = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=DEVICE)
        cell = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=DEVICE)
        embedded = self.embedding(input_seqs)
        embedded = self.dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)

        if self.rnn_type == "GRU":
            self.gru.flatten_parameters()
            output, hidden = self.gru(packed, hidden)
        else:
            self.lstm.flatten_parameters()
            output, (hidden, cell) = self.lstm(packed, (hidden, cell))
        # print("output", output.data.size())
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=self.max_length)  # unpack (back to padded)
        if self.num_directions == 2:
            hidden = self.convert(hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
                                  .transpose(1, 3)).transpose(1,3).squeeze(1)
        return output, hidden, cell

    def init_rnn(self):
        if self.rnn_type == "GRU":
            for name, param in self.gru.named_parameters():
                if 'bias' or 'weight' in name:
                    nn.init.uniform_(param, -0.1, 0.1)
        else:
            for name, param in self.lstm.named_parameters():
                if 'bias' or 'weight' in name:
                    nn.init.uniform_(param, -0.1, 0.1)


class BatchDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=4, dropout_p=0.1, max_length=MAX_LENGTH, span_size=SPAN_SIZE, rnn_type="GRU", num_directions=1):
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
            for name, param in self.gru.named_parameters():
                if 'bias' or 'weight' in name:
                    nn.init.uniform(param, -0.1, 0.1)

        else:
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout_p,
                                batch_first=True)
            for name, param in self.lstm.named_parameters():
                if 'bias' or 'weight' in name:
                    nn.init.uniform(param, -0.1, 0.1)
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
        # print("embeddes", embeddeds.size())

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

        # output = F.relu(rnn_output)
        output = self.out(rnn_output)
        output = F.log_softmax(output, dim=2)

        attn_weight = 0

        return output, hidden, cell, attn_weight


class BatchKspanDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=4, dropout_p=0.1, max_length=MAX_LENGTH, span_size=SPAN_SIZE, rnn_type="GRU", num_directions=1):
        super(BatchKspanDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.span_size = span_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.cat_embeddings = nn.Linear(self.hidden_size * self.span_size, self.hidden_size)
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
        embeddeds = self.embedding(inputs)  # B x S -> B x S x H
        embeddeds = embeddeds.view(bsz, -1)  # B x (S x H)
        embeddeds = self.dropout(embeddeds)  # B x (S x H)

        embeddeds = self.cat_embeddings(embeddeds).unsqueeze(1)
        embeddeds = F.relu(embeddeds)

        if self.rnn_type == "GRU":
            self.gru.flatten_parameters()
            rnn_output, hidden = self.gru(embeddeds, hidden)
        else:
            self.lstm.flatten_parameters()
            rnn_output, (hidden, cell) = self.lstm(embeddeds, (hidden, cell))

        output = self.out(rnn_output).view(bsz, self.span_size, -1)
        output = F.log_softmax(output, dim=2)

        attn_weight = 0

        return output, hidden, cell, attn_weight


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=4, dropout_p=0.1, max_length=MAX_LENGTH, span_size=SPAN_SIZE, rnn_type="GRU", num_directions=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout_p

        self.embedding = nn.Embedding(output_size, hidden_size)

        if rnn_type == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.out = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout_p)
        self.rnn_type = rnn_type

    def forward(self, inputs, hidden, cell, encoder_outputs):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # context = [n layers * n directions, batch size, hid dim]

        # n layers and n directions in the decoder will both always be 1, therefore:
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim]


        # input = [1, batch size]

        embedded = self.dropout(self.embedding(inputs))

        # embedded = [1, batch size, emb dim]


        # emb_con = [1, batch size, emb dim + hid dim]

        if self.rnn_type == "GRU":
            output, hidden = self.rnn(embedded, hidden)
        else:
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        # sent len, n layers and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        # output = [batch size, emb dim + hid dim * 2]

        output = self.out(output)
        output = F.log_softmax(output, dim=2)

        # prediction = [batch size, output dim]

        attn_weight = 0

        return output, hidden, cell, attn_weight


class BatchBahdanauAttnKspanDecoderRNN2(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=4, dropout_p=0.1, max_length=MAX_LENGTH, span_size=SPAN_SIZE,
                 rnn_type="GRU", num_directions=1):
        super(BatchBahdanauAttnKspanDecoderRNN2, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.span_size = span_size
        self.rnn_type = rnn_type
        self.num_directions = num_directions

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.cat_embeddings = nn.Linear(self.hidden_size * self.span_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * (num_directions + 1), self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        if rnn_type == "GRU":
            self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout_p, batch_first=True)
        else:
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout_p, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size * span_size)

    def forward(self, inputs, hidden, cell, encoder_outputs):
        # Assume inputs is padded to max length, max_length is multiple of span_size
        # ==========================================================================

        bsz = inputs.size()[0]
        embeddeds = self.embedding(inputs)  # B x S -> B x S x H
        embeddeds = embeddeds.view(bsz, -1)  # B x (S x H)
        embeddeds = self.dropout(embeddeds)  # B x (S x H)

        embeddeds = self.cat_embeddings(embeddeds)
        attn_weights = F.softmax(
            self.attn(torch.cat((embeddeds, hidden[-1]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)
        embeddeds = torch.cat((embeddeds.unsqueeze(1), attn_applied), 2)
        embeddeds = self.attn_combine(embeddeds)

        embeddeds = F.relu(embeddeds)

        if self.rnn_type == "GRU":
            self.gru.flatten_parameters()
            rnn_output, hidden = self.gru(embeddeds, hidden)
        else:
            self.lstm.flatten_parameters()
            rnn_output, (hidden, cell) = self.lstm(embeddeds, (hidden, cell))

        output = self.out(rnn_output).view(bsz, self.span_size, -1)
        output = F.log_softmax(output, dim=2)

        return output, hidden, cell, attn_weights

    def init_rnn(self):
        if self.rnn_type =="GRU":
            for name, param in self.gru.named_parameters():
                if 'bias' or 'weight' in name:
                    nn.init.uniform_(param, -0.1, 0.1)
        else:
            for name, param in self.lstm.named_parameters():
                if 'bias' or 'weight' in name:
                    nn.init.uniform_(param, -0.1, 0.1)


class BatchBahdanauAttnKspanDecoderRNN3(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=4, dropout_p=0.1, max_length=MAX_LENGTH, span_size=SPAN_SIZE,
                 rnn_type="GRU", num_directions=1):
        super(BatchBahdanauAttnKspanDecoderRNN3, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.span_size = span_size
        self.rnn_type = rnn_type
        self.num_directions = num_directions

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.cat_embeddings = nn.Linear(self.hidden_size * self.span_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * (num_directions + 1), self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        if rnn_type == "GRU":
            self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout_p, batch_first=True)
        else:
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout_p, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size * span_size)

    def forward(self, inputs, hidden, cell, encoder_outputs):
        # Assume inputs is padded to max length, max_length is multiple of span_size
        # ==========================================================================

        bsz = inputs.size()[0]
        embeddeds = self.embedding(inputs)  # B x S -> B x S x H
        embeddeds = embeddeds.view(bsz, -1)  # B x (S x H)
        embeddeds = self.dropout(embeddeds)  # B x (S x H)

        embeddeds = self.cat_embeddings(embeddeds)
        attn_weights = F.softmax(
            self.attn(torch.cat((embeddeds, hidden[-1]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)
        embeddeds = torch.cat((embeddeds.unsqueeze(1), attn_applied), 2)
        embeddeds = self.attn_combine(embeddeds)

        embeddeds = F.relu(embeddeds)

        if self.rnn_type == "GRU":
            self.gru.flatten_parameters()
            rnn_output, hidden = self.gru(embeddeds, hidden)
        else:
            self.lstm.flatten_parameters()
            rnn_output, (hidden, cell) = self.lstm(embeddeds, (hidden, cell))

        output = self.dropout2(rnn_output)
        output = self.out(output).view(bsz, self.span_size, -1)
        output = F.log_softmax(output, dim=2)

        return output, hidden, cell, attn_weights

    def init_rnn(self):
        if self.rnn_type =="GRU":
            for name, param in self.gru.named_parameters():
                if 'bias' or 'weight' in name:
                    nn.init.uniform_(param, -0.1, 0.1)
        else:
            for name, param in self.lstm.named_parameters():
                if 'bias' or 'weight' in name:
                    nn.init.uniform_(param, -0.1, 0.1)

