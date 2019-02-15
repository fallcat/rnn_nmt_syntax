from .preprocess2 import *

import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
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
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
        # return [torch.zeros(1, 1, self.hidden_size, device=device) for _ in range(self.num_layers)]


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
        return torch.zeros(1, 1, self.hidden_size, device=device)


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
        return torch.zeros(1, 1, self.hidden_size, device=device)


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
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)


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
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
