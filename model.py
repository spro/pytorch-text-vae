import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from helpers import *

# Encoder
# ------------------------------------------------------------------------------

# Encode into Z with mu and log_var

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(input_size, hidden_size)
        self.e2o = nn.Linear(hidden_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.o2m = nn.Linear(hidden_size, output_size)
        self.o2l = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        embedded = self.embed(input)
        embedded = self.e2o(embedded).unsqueeze(1)

        hidden = self.init_hidden()
        output, hidden = self.gru(embedded, hidden)
        output = output[-1] # Take only the last value

        mu = self.o2m(output)
        logvar = self.o2l(output)
        z = self.sample(mu, logvar)
        return mu, logvar, z

    def sample(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        if USE_CUDA: eps = eps.cuda()
        return eps.mul(std).add_(mu)

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden

# Decoder
# ------------------------------------------------------------------------------

# Decode from Z into sequence, regular LM

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout_p=0.05):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, hidden_size)
        self.z2h = nn.Linear(input_size, hidden_size * n_layers)
        self.dropout = nn.Dropout(dropout_p)
        # self.gru = nn.GRU(hidden_size + input_size, hidden_size, n_layers)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, z, inputs):
        n_steps = inputs.size(0)
        outputs = Variable(torch.zeros(n_steps + 1, 1, self.output_size))
        if USE_CUDA: outputs = outputs.cuda()

        # Forward SOS through without dropout
        sos_input = Variable(torch.LongTensor([SOS]))
        if USE_CUDA: sos_input = sos_input.cuda()
        output, hidden = self.step(z, sos_input)
        outputs[0] = output

        for i in range(n_steps): # Before EOS
            output, hidden = self.step(z, inputs[i], hidden, True)
            outputs[i + 1] = output
        return outputs.squeeze(1)

    def sample(self, z, n_steps):
        outputs = Variable(torch.zeros(n_steps + 1, 1, self.output_size))
        if USE_CUDA: outputs = outputs.cuda()

        sos_input = input = Variable(torch.LongTensor([SOS]))
        if USE_CUDA: sos_input = sos_input.cuda()
        output, hidden = self.step(z, sos_input)
        outputs[0] = output
        top_i = output.data.topk(1)[1][0][0]
        input = Variable(torch.LongTensor([top_i]))
        if USE_CUDA: input = input.cuda()

        for i in range(n_steps):
            output, hidden = self.step(z, input, hidden, True)
            outputs[i + 1] = output

            # Sample top value only
            top_i = output.data.topk(1)[1][0][0]

            # Sample from the network as a multinomial distribution
            # output_dist = output.data.view(-1).div(temperature).exp()
            # top_i = torch.multinomial(output_dist, 1)[0]

            if top_i == EOS: break
            input = Variable(torch.LongTensor([top_i]))
            if USE_CUDA: input = input.cuda()
        return outputs.squeeze(1)

    def step(self, z, input, hidden=None, dropout=False):
        # print('[DecoderRNN.step]', 'z =', z.size(), 'i =', input.size(), 'h =', hidden.size())
        input = self.embed(input)
        if dropout:
            input = self.dropout(input)
        if hidden is None:
            hidden = self.z2h(z).view(self.n_layers, 1, self.hidden_size)
            # print('hidden size', hidden.size())
        # input = torch.cat((z, input), 1).unsqueeze(0)
        input = input.unsqueeze(0)
        output, hidden = self.gru(input, hidden)
        output = self.out(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden

# Container
# ------------------------------------------------------------------------------

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        m, l, z = self.encoder(input)
        decoded = self.decoder(z, input)
        return m, l, z, decoded

