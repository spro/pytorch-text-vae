import sconce
import sys

from model import *

hidden_size = 500
embed_size = 50
learning_rate = 0.0001
n_epochs = 100000
grad_clip = 1.0

kld_start_inc = 10000
kld_weight = 0.05
kld_max = 0.1
kld_inc = 0.000002
temperature = 0.9
temperature_min = 0.5
temperature_dec = 0.000002

# Training
# ------------------------------------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python train.py [filename]")
    sys.exit(1)

file, file_len = read_file(sys.argv[1])
# file, file_len = read_file('../practical-pytorch/data/first-names.txt')

lines = [line.strip() for line in file.split('\n')]
print('n lines', len(lines))

def good_size(line):
    return len(line) >= MIN_LENGTH and len(line) <= MAX_LENGTH

def good_content(line):
    return 'http' not in line and '/' not in line

lines = [line for line in lines if good_size(line) and good_content(line)]
print('n lines', len(lines))
random.shuffle(lines)

def random_training_set():
    line = random.choice(lines)
    inp = char_tensor(line)
    target = char_tensor(line, True)
    return inp, target

e = EncoderRNN(n_characters, hidden_size, embed_size)
d = DecoderRNN(embed_size, hidden_size, n_characters, 2)
vae = VAE(e, d)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()

if USE_CUDA:
    vae.cuda()
    criterion.cuda()

log_every = 200
save_every = 5000
job = sconce.Job('vae', {
    'hidden_size': hidden_size,
    'embed_size': embed_size,
    'learning_rate': learning_rate,
    'kld_weight': kld_weight,
    'temperature': temperature,
    'grad_clip': grad_clip,
})

job.log_every = log_every

def save():
    save_filename = 'vae.pt'
    torch.save(vae, save_filename)
    print('Saved as %s' % save_filename)

try:
    for epoch in range(n_epochs):
        input, target = random_training_set()

        optimizer.zero_grad()

        m, l, z, decoded = vae(input, temperature)
        if temperature > temperature_min:
            temperature -= temperature_dec

        loss = criterion(decoded, target)
        job.record(epoch, loss.data[0])

        KLD = (-0.5 * torch.sum(l - torch.pow(m, 2) - torch.exp(l) + 1, 1)).mean().squeeze()
        loss += KLD * kld_weight

        if epoch > kld_start_inc and kld_weight < kld_max:
            kld_weight += kld_inc

        loss.backward()
        # print('from', next(vae.parameters()).grad.data[0][0])
        ec = torch.nn.utils.clip_grad_norm(vae.parameters(), grad_clip)
        # print('to  ', next(vae.parameters()).grad.data[0][0])
        optimizer.step()

        if epoch % log_every == 0:
            print('[%d] %.4f (k=%.4f, t=%.4f, kl=%.4f, ec=%.4f)' % (
                epoch, loss.data[0], kld_weight, temperature, KLD.data[0], ec
            ))
            print('   (target) "%s"' % longtensor_to_string(target))
            generated = vae.decoder.generate(z, MAX_LENGTH, temperature)
            print('(generated) "%s"' % tensor_to_string(generated))
            print('')

        if epoch > 0 and epoch % save_every == 0:
            save()

    save()

except KeyboardInterrupt as err:
    print("ERROR", err)
    print("Saving before quit...")
    save()

