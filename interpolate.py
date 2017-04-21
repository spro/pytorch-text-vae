from model import *

vae = torch.load('vae.pt')

def random_samples(m, l):
    for i in range(n_samples):
        m_jitter = torch.FloatTensor(m.size()).normal_()
        l_jitter = torch.FloatTensor(l.size()).normal_()
        z = vae.encoder.sample(m + m_jitter, l + l_jitter)
        print('(?)', tensor_to_string(vae.decoder.sample(z, 20)))

m, l, z0, d0 = vae(char_tensor('haha'))
print('(d0)', tensor_to_string(d0))
# random_samples()
# print(m, l)

m, l, z1, d1 = vae(char_tensor('ok'))
print('(d1)', tensor_to_string(d1))
# random_samples()

rm = Variable(torch.FloatTensor(m.size()).normal_())
rl = Variable(torch.FloatTensor(l.size()).normal_())
z = vae.encoder.sample(m, l)
z = vae.encoder.sample(rm, rl)
print('(random)', tensor_to_string(vae.decoder.sample(z,  20)))

diff = z1 - z0
n_samples = 10

for i in range(n_samples + 1):
    p = i / n_samples
    print('(?)', tensor_to_string(vae.decoder.sample(z0 + diff * p, 20)))

