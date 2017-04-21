import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from helpers import *
import sconce
import sys

def sample(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = torch.FloatTensor(std.size()).normal_()
    eps = Variable(eps)
    return eps.mul(std).add_(mu)

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
        # output = self.out(output.squeeze(1))
        mu = self.o2m(output)
        logvar = self.o2l(output)
        # mu = output[:self.output_size]
        # logvar = output[self.output_size:]
        z = sample(mu, logvar)
        return mu, logvar, z

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

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

        # Forward SOS through without dropout
        sos_input = Variable(torch.LongTensor([SOS]))
        output, hidden = self.step(z, sos_input)
        outputs[0] = output

        for i in range(n_steps): # Before EOS
            output, hidden = self.step(z, inputs[i], hidden, True)
            outputs[i + 1] = output
        return outputs.squeeze(1)

    def sample(self, z, n_steps):
        outputs = Variable(torch.zeros(n_steps + 1, 1, self.output_size))

        sos_input = input = Variable(torch.LongTensor([SOS]))
        output, hidden = self.step(z, sos_input)
        outputs[0] = output
        top_i = output.data.topk(1)[1][0][0]
        input = Variable(torch.LongTensor([top_i]))

        for i in range(n_steps):
            output, hidden = self.step(z, input, hidden, True)
            outputs[i + 1] = output

            # top_i = output.data.topk(1)[1][0][0]
            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            if top_i == EOS: break
            input = Variable(torch.LongTensor([top_i]))
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
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

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

# Training
# ------------------------------------------------------------------------------

file, file_len = read_file('../nexus/chats-chad.txt')
# file, file_len = read_file('../practical-pytorch/data/first-names.txt')

lines = [line.strip() for line in file.split('\n')]
print('n lines', len(lines))
def good_size(line): return len(line) > 1 and len(line) < 50
def good_content(line): return 'http' not in line and '/' not in line
lines = list(filter(good_size, lines))
lines = list(filter(good_content, lines))
print('n lines', len(lines))
random.shuffle(lines)

def random_training_set():
    line = random.choice(lines)
    inp = char_tensor(line)
    target = char_tensor(line, True)
    return inp, target

hidden_size = 200
embed_size = 100
learning_rate = 0.0005
n_epochs = 100000

e = EncoderRNN(n_characters, hidden_size, embed_size)
d = DecoderRNN(embed_size, hidden_size, n_characters, 2)
vae = VAE(e, d)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

job = sconce.Job('vae')
log_every = 100
job.log_every = log_every

kld_weight = 0.01
temperature = 0.5

for epoch in range(n_epochs):
    input, target = random_training_set()

    optimizer.zero_grad()

    m, l, z, decoded = vae(input)

    loss = criterion(decoded, target)

    if epoch > 10000:
        KLD = (-0.5 * torch.sum(l - torch.pow(m, 2) - torch.exp(l) + 1, 1)).mean().squeeze()
        KLD /= decoded.size(0)
        loss += KLD * kld_weight

        if kld_weight < 1:
            kld_weight += 0.00001

    loss.backward()
    optimizer.step()
    job.record(epoch, loss.data[0])

    if epoch % log_every == 0:
        print('[%d] %.4f (%.4f)' % (epoch, loss.data[0], kld_weight))
        print('    (input) "%s"' % longtensor_to_string(input))
        print('  (decoded) "%s"' % tensor_to_string(decoded))
        sampled = vae.decoder.sample(z, input.size(0))
        print('  (sampled) "%s"' % tensor_to_string(sampled))
        print('')

def save():
    save_filename = 'vae.pt'
    torch.save(vae, save_filename)
    print('Saved as %s' % save_filename)

save()
