import sconce
import sys

from model import *

# Training
# ------------------------------------------------------------------------------

file, file_len = read_file('./chats-chad.txt')
# file, file_len = read_file('../practical-pytorch/data/first-names.txt')

lines = [line.strip() for line in file.split('\n')]
print('n lines', len(lines))
def good_size(line): return len(line) > 1 and len(line) < MAX_LENGTH
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

hidden_size = 500
embed_size = 100
learning_rate = 0.0001
n_epochs = 50000
grad_clip = 0.5

e = EncoderCNN(n_characters, hidden_size, embed_size)
d = DecoderRNN(embed_size, hidden_size, n_characters, 2)
vae = VAE(e, d)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()

if USE_CUDA:
    vae.cuda()
    criterion.cuda()

log_every = 200
save_every = 5000
job = sconce.Job('vae')
job.log_every = log_every

kld_weight = 0.01

def save():
    save_filename = 'vae.pt'
    torch.save(vae, save_filename)
    print('Saved as %s' % save_filename)

try:
    for epoch in range(n_epochs):
        input, target = random_training_set()

        optimizer.zero_grad()

        m, l, z, decoded = vae(input)

        loss = criterion(decoded, target)
        job.record(epoch, loss.data[0])

        KLD = (-0.5 * torch.sum(l - torch.pow(m, 2) - torch.exp(l) + 1, 1)).mean().squeeze()
        loss += KLD * kld_weight

        if epoch > 20000 and kld_weight < 0.03:
            kld_weight += 0.0000001

        loss.backward()
        torch.nn.utils.clip_grad_norm(vae.parameters(), grad_clip)
        optimizer.step()

        if epoch % log_every == 0:
            print('[%d] %.4f (%.4f)' % (epoch, loss.data[0], kld_weight))
            print('   (target) "%s"' % longtensor_to_string(target))
            sampled = vae.decoder.sample(z, MAX_LENGTH)
            print('  (sampled) "%s"' % tensor_to_string(sampled))
            print('')

        if epoch > 0 and epoch % save_every == 0:
            save()

    save()

except KeyboardInterrupt as err:
    print("ERROR", err)
    print("Saving before quit...")
    save()

