class EncoderCNN(nn.Module):
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
        z = reparametrize(mu, logvar)
        return mu, logvar, z


class DecoderCNN(nn.Module):
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

