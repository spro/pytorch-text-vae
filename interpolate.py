from model import *

vae = torch.load('vae.pt')
vae.train(False)

TEMPERATURE = 0.01
N_SAMPLES = 10
N_STEPS = 10

def random_sample():
    size = vae.encoder.output_size
    rm = Variable(torch.FloatTensor(1, size).normal_())
    rl = Variable(torch.FloatTensor(1, size).normal_())
    if USE_CUDA:
        rm = rm.cuda()
        rl = rl.cuda()
    z = vae.encoder.sample(rm, rl)
    return z

for s in range(1, N_SAMPLES):
    z0 = random_sample()
    z1 = random_sample()
    diff = z1 - z0

    last_s = ''

    print('(z0)', tensor_to_string(vae.decoder.generate(z0,  MAX_LENGTH, TEMPERATURE)))

    for i in range(1, N_STEPS):
        p = i * 1.0 / N_STEPS
        s = tensor_to_string(vae.decoder.generate(z0 + diff * p, MAX_LENGTH, TEMPERATURE))
        if last_s != s:
            print('  .)', s)
        last_s = s

    print('(z1)', tensor_to_string(vae.decoder.generate(z1,  MAX_LENGTH, TEMPERATURE)))
    print('\n')

