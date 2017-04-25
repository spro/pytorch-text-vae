from model import *

vae = torch.load('vae.pt')

def random_sample():
    size = 100
    rm = Variable(torch.FloatTensor(1, size).normal_())
    rl = Variable(torch.FloatTensor(1, size).normal_())
    if USE_CUDA:
        rm = rm.cuda()
        rl = rl.cuda()
    z = vae.encoder.sample(rm, rl)
    return z

z0 = random_sample()
z1 = random_sample()

diff = z1 - z0
n_samples = 10

last_s = ''

print('(z0)', tensor_to_string(vae.decoder.sample(z0,  MAX_LENGTH)))

for i in range(1, n_samples):
    p = i * 1.0 / n_samples
    s = tensor_to_string(vae.decoder.sample(z0 + diff * p, MAX_LENGTH))
    if last_s != s:
        print('  .)', s)
    last_s = s

print('(z1)', tensor_to_string(vae.decoder.sample(z1,  MAX_LENGTH)))

