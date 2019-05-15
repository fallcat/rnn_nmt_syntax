import torch
import torch.nn as nn
import torch.nn.functional as F
from model import PAD_token, SOS_token, EOS_token, MAX_LENGTH, SPAN_SIZE, DEVICE


class RNMTPlusEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_p=0.1, rnn_type="GRU", num_directions=1):
        super(RNMTPlusEncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)
        self.convert = nn.Linear(2, 1)

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn_type = rnn_type
        args = [hidden_size, dropout_p, rnn_type, num_directions]
        self.encoder_layers = nn.ModuleList([
            RNMTPlusEncoderLayer(*args)
            for _ in range(num_layers)
        ])
        self.projection = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input_seqs, input_lengths, total_length, hidden=None):
        embedded = self.embedding(input_seqs)
        output = self.dropout(embedded)

        for encoder_layer in self.encoder_layers:
            output, hidden, cell = encoder_layer(output, input_lengths)

        output = self.projection(output)

        return output, hidden, cell


class RNMTPlusEncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout_p=0.1, rnn_type="GRU", num_directions=1):
        super(RNMTPlusEncoderLayer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = 1
        self.num_directions = num_directions
        self.dropout = nn.Dropout(dropout_p)
        if num_directions == 2:
            self.convert = nn.Linear(2, 1)

        self.rnn_type = rnn_type
        if rnn_type == "GRU":
            self.gru = nn.GRU(hidden_size, hidden_size, self.num_layers, batch_first=True,
                              bidirectional=(num_directions == 2))
        else:
            self.lstm = nn.LSTM(hidden_size, hidden_size, self.num_layers, batch_first=True,
                                bidirectional=(num_directions == 2))

        self.layer_norm = nn.LayerNorm(self.num_directions * self.hidden_size)

    def forward(self, inputs, input_lengths):
        batch_size = inputs.size()[0]
        input_length = inputs.size()[1]

        hidden = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=DEVICE)
        cell = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=DEVICE)

        packed = torch.nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True)

        if self.rnn_type == "GRU":
            self.gru.flatten_parameters()
            output, hidden = self.gru(packed, hidden)
        else:
            self.lstm.flatten_parameters()
            output, (hidden, cell) = self.lstm(packed, (hidden, cell))
        # print("output", output.data.size())
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=input_length)  # unpack (back to padded)
        output = self.layer_norm(output)
        if self.num_directions == 2:
            hidden = self.convert(hidden.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
                                  .transpose(1, 3)).transpose(1,3).squeeze(1)
            output = self.convert(output.view(batch_size, -1, self.num_directions, self.hidden_size)
                                  .transpose(2, 3)).squeeze(3)
        output = self.dropout(output)
        output = inputs + output
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


class RNMTPlusDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=4, dropout_p=0.1, span_size=SPAN_SIZE,
                 rnn_type="GRU", num_directions=1, num_heads=4):
        super(RNMTPlusDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.span_size = span_size
        self.rnn_type = rnn_type
        self.num_directions = num_directions
        self.num_heads = num_heads

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.cat_embeddings = nn.Linear(self.hidden_size * self.span_size, self.hidden_size)
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, self.num_heads)
        args = [hidden_size, dropout_p, rnn_type, num_heads]
        self.decoder_layers = nn.ModuleList([
            RNMTPlusDecoderLayer2(*args)
            for _ in range(num_layers)
        ])
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        if rnn_type == "GRU":
            self.gru = nn.GRU(self.hidden_size, self.hidden_size, 1, dropout=self.dropout_p, batch_first=True)
        else:
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, 1, dropout=self.dropout_p, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size * span_size)

    def forward(self, inputs, hiddens, cells, encoder_outputs):
        # Assume inputs is padded to max length, max_length is multiple of span_size
        # ==========================================================================

        bsz = inputs.size()[0]
        embeddeds = self.embedding(inputs)  # B x S -> B x S x H
        embeddeds = embeddeds.view(bsz, -1)  # B x (S x H)
        embeddeds = self.dropout(embeddeds)  # B x (S x H)

        embeddeds = self.cat_embeddings(embeddeds).unsqueeze(1)

        if self.rnn_type == "GRU":
            self.gru.flatten_parameters()
            rnn_output, hiddens[0] = self.gru(embeddeds, hiddens[0].unsqueeze(0))
        else:
            self.lstm.flatten_parameters()
            rnn_output, (hiddens[0], cells[0]) = self.lstm(embeddeds, (hiddens[0].unsqueeze(0), cells[0].unsqueeze(0)))

        attn_output, attn_output_weights = self.multihead_attn(rnn_output.transpose(0, 1),
                                                               encoder_outputs.transpose(0, 1),
                                                               encoder_outputs.transpose(0, 1))

        attn_output = attn_output.transpose(0, 1)
        for i, decoder_layer in enumerate(self.decoder_layers):
            rnn_output, hiddens[i+1], cells[i+1] = decoder_layer(rnn_output, hiddens[i+1], cells[i+1], attn_output)

        output = torch.cat((rnn_output, attn_output), 2)
        output = self.attn_combine(output)
        output = self.out(output).view(bsz, self.span_size, -1)
        output = F.log_softmax(output, dim=2)

        return output, hiddens, cells, attn_output_weights

    def init_rnn(self):
        if self.rnn_type =="GRU":
            for name, param in self.gru.named_parameters():
                if 'bias' or 'weight' in name:
                    nn.init.uniform_(param, -0.1, 0.1)
        else:
            for name, param in self.lstm.named_parameters():
                if 'bias' or 'weight' in name:
                    nn.init.uniform_(param, -0.1, 0.1)


class RNMTPlusDecoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout_p=0.1, rnn_type="GRU", num_heads=4):
        super(RNMTPlusDecoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.dropout_p = dropout_p
        self.rnn_type = rnn_type
        self.num_directions = 1
        self.num_heads = num_heads

        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, self.num_heads)
        self.dropout = nn.Dropout(self.dropout_p)
        if rnn_type == "GRU":
            self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout_p, batch_first=True)
        else:
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout_p, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.num_directions * self.hidden_size)

    def forward(self, inputs, hidden, cell, attn_outputs):
        # Assume inputs is padded to max length, max_length is multiple of span_size
        # ==========================================================================
        embeddeds = torch.cat((inputs, attn_outputs), 2)
        embeddeds = self.attn_combine(embeddeds)

        if self.rnn_type == "GRU":
            self.gru.flatten_parameters()
            rnn_output, hidden = self.gru(embeddeds, hidden)
        else:
            self.lstm.flatten_parameters()
            rnn_output, (hidden, cell) = self.lstm(embeddeds, (hidden, cell))

        output = self.layer_norm(rnn_output)
        output = self.dropout(output)
        output = output + inputs
        return output, hidden, cell

    def init_rnn(self):
        if self.rnn_type =="GRU":
            for name, param in self.gru.named_parameters():
                if 'bias' or 'weight' in name:
                    nn.init.uniform_(param, -0.1, 0.1)
        else:
            for name, param in self.lstm.named_parameters():
                if 'bias' or 'weight' in name:
                    nn.init.uniform_(param, -0.1, 0.1)


class RNMTPlusDecoderLayer2(nn.Module):
    def __init__(self, hidden_size, dropout_p=0.1, rnn_type="GRU", num_heads=4):
        super(RNMTPlusDecoderLayer2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.dropout_p = dropout_p
        self.rnn_type = rnn_type
        self.num_directions = 1
        self.num_heads = num_heads

        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, self.num_heads)
        self.dropout = nn.Dropout(self.dropout_p)
        if rnn_type == "GRU":
            self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout_p, batch_first=True)
        else:
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout_p, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.num_directions * self.hidden_size)

    def forward(self, inputs, hidden, cell, attn_outputs):
        # Assume inputs is padded to max length, max_length is multiple of span_size
        # ==========================================================================
        embeddeds = torch.cat((inputs, attn_outputs), 2)
        embeddeds = self.attn_combine(embeddeds)

        if self.rnn_type == "GRU":
            self.gru.flatten_parameters()
            rnn_output, hidden = self.gru(embeddeds, hidden.unsqueeze(0))
        else:
            self.lstm.flatten_parameters()
            rnn_output, (hidden, cell) = self.lstm(embeddeds, (hidden.unsqueeze(0), cell.unsqueeze(0)))

        output = self.layer_norm(rnn_output)
        output = self.dropout(output)
        output = output + inputs
        return output, hidden.squeeze(0), cell.squeeze(0)

    def init_rnn(self):
        if self.rnn_type =="GRU":
            for name, param in self.gru.named_parameters():
                if 'bias' or 'weight' in name:
                    nn.init.uniform_(param, -0.1, 0.1)
        else:
            for name, param in self.lstm.named_parameters():
                if 'bias' or 'weight' in name:
                    nn.init.uniform_(param, -0.1, 0.1)

