import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from helpers import *

MAX_LENGTH = 50
MAX_SAMPLE = False
TEMPERATURE = 0.5

class RNN:
    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden

class Encoder(nn.Module):
    def sample(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        if USE_CUDA: eps = eps.cuda()
        return eps.mul(std).add_(mu)

class Decoder(nn.Module):
    def sample(self, z, n_steps):
        pass

    def output_to_input(self, output, temperature=TEMPERATURE):
        if MAX_SAMPLE:
            # Sample top value only
            top_i = output.data.topk(1)[1][0][0]

        else:
            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

        input = Variable(torch.LongTensor([top_i]))
        if USE_CUDA: input = input.cuda()
        return input, top_i

    def forward(self, z, inputs):
        n_steps = inputs.size(0)
        outputs = Variable(torch.zeros(n_steps + 1, 1, self.output_size))
        if USE_CUDA: outputs = outputs.cuda()

        sos_tensor = Variable(torch.LongTensor([SOS]))
        output, hidden = self.step(0, z, sos_tensor)
        outputs[0] = output

        for i in range(n_steps):
            output, hidden = self.step(i, z, inputs[i], hidden, True)
            outputs[i + 1] = output

        return outputs.squeeze(1)

    def sample(self, z, n_steps, use_dropout=True):
        outputs = Variable(torch.zeros(n_steps + 1, 1, self.output_size))
        if USE_CUDA: outputs = outputs.cuda()

        sos_tensor = Variable(torch.LongTensor([SOS]))
        output, hidden = self.step(0, z, sos_tensor)
        input, top_i = self.output_to_input(output)
        outputs[0] = output

        for i in range(1, n_steps + 1):
            output, hidden = self.step(i, z, input, hidden, use_dropout)
            outputs[i] = output
            input, top_i = self.output_to_input(output)
            if top_i == EOS: break

        return outputs.squeeze(1)

# Encoder
# ------------------------------------------------------------------------------

# Encode into Z with mu and log_var

class EncoderRNN(Encoder, RNN):
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

class EncoderCNN(Encoder):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(EncoderCNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(input_size, hidden_size)

        self.c1 = nn.Conv1d(hidden_size, 100, 2)
        self.p1 = nn.MaxPool1d(2)
        self.c2 = nn.Conv1d(100, hidden_size, 2)
        self.p2 = nn.MaxPool1d(3)
        convolved_size = self.hidden_size * 7

        self.o2m = nn.Linear(convolved_size, output_size)
        self.o2l = nn.Linear(convolved_size, output_size)

    def forward(self, input):
        # print('\n[EncoderCNN.forward]')

        input = self.embed(input)

        input_padded = Variable(torch.zeros(MAX_LENGTH, self.hidden_size))
        input_padded[:input.size(0)] = input
        input = input_padded.transpose(0, 1)
        input = input.unsqueeze(0)

        input = self.c1(input)
        input = self.p1(input)
        # print('(c1 p1) input', input.size())

        input = self.c2(input)
        input = self.p2(input)
        # print('(c2 p2) input', input.size())

        output = input.view(1, -1)
        # print('output', output.size())

        mu = self.o2m(output)
        logvar = self.o2l(output)
        z = self.sample(mu, logvar)
        return mu, logvar, z

# Decoder
# ------------------------------------------------------------------------------

# Decode from Z into sequence, regular LM

class DecoderCNN(Decoder):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout_p=0.05):
        super(DecoderCNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embed = nn.Embedding(output_size, hidden_size)
        self.uc1 = nn.ConvTranspose1d(input_size, 200, 15)
        self.uc2 = nn.ConvTranspose1d(200, hidden_size, 15)
        self.uc3 = nn.ConvTranspose1d(hidden_size, hidden_size, 13)
        self.uc4 = nn.ConvTranspose1d(hidden_size, output_size, 11)
        self.gru = nn.GRU(input_size + hidden_size, output_size)

    def dconv(self, z, inputs):

        # print('\n[DecoderCNN.forward]')
        # print('outputs', outputs.size())

        z = z.transpose(0, 1)
        # print('         z =', z.size())
        # print('    inputs =', inputs.size())

        u = self.uc1(z.unsqueeze(0))
        # print('         u1=', u.size())
        u = self.uc2(u)
        # print('         u2=', u.size())
        u = self.uc3(u)
        # print('         u3=', u.size())
        u = self.uc4(u)
        # print('         u4=', u.size())

        # u = u.transpose(1, 2).transpose(0, 1)
        u = u.squeeze(0).transpose(0, 1)
        # u = u[:n_steps + 1]
        # print('         u =', u.size())
        return u

    def step(self, s, u, input, hidden=None, test=False):
        u = u.unsqueeze(0)
        # print('u = ', u.size())
        # print('input = ', input.size())
        input = self.embed(input)
        # print('input = ', input.size())
        input = input.unsqueeze(0)
        # print('u :', u.size())
        # print('input :', input.size())
        inp = torch.cat((u, input), 2)
        # print('inp :', inp.size())
        return self.gru(inp, hidden)

class DecoderRNN(Decoder, RNN):
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

    def step(self, s, z, input=None, hidden=None, use_dropout=False):
        # print('[DecoderRNN.step] s =', s, 'z =', z.size(), 'i =', input.size(), 'h =', hidden.size())
        if s == 0:
            # Forward SOS through without dropout
            hidden = self.z2h(z).view(self.n_layers, 1, self.hidden_size)
            use_dropout = False
        input = self.embed(input)
        if use_dropout: input = self.dropout(input)
        input = input.unsqueeze(0)
        output, hidden = self.gru(input, hidden)
        output = self.out(output.view(1, -1))
        return output, hidden

# Container
# ------------------------------------------------------------------------------

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs):
        m, l, z = self.encoder(inputs)
        decoded = self.decoder(z, inputs)
        return m, l, z, decoded

# Test

if __name__ == '__main__':
    hidden_size = 200
    embed_size = 100
    e = EncoderCNN(n_characters, hidden_size, embed_size)
    d = DecoderCNN(embed_size, hidden_size, n_characters, 2)
    vae = VAE(e, d)
    m, l, z, decoded = vae(char_tensor('@spro'))
    print('m =', m.size())
    print('l =', l.size())
    print('z =', z.size())
    print('decoded', tensor_to_string(decoded))

