from models import get_model, VAE

import pytorch_lightning as pl
from tokenizer import SmilesTokenizer
import sys
from configs import configs
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--s1", help='s1')
parser.add_argument("--s2", help='s2')
parser.add_argument("--epsilon", help='epsilon', default=0.1)
parser.add_argument("--n", help='number of mol to sample')
args = parser.parse_args()

tokenizer = SmilesTokenizer.load('./a.vocab')

model = get_model('trans', configs)
assert isinstance(model, pl.LightningModule)

is_gpu = True
model.load('./transMol-transMol/a.ckpt')
if is_gpu:
    model = model.cuda()
s1 = args.s1
s2 = args.s2
epsilon = float(args.epsilon)
z1 = model.encode_latent(s1, tokenizer, configs['max_len'], is_gpu)

z2 = model.encode_latent(s2, tokenizer, configs['max_len'], is_gpu)
#n = int(args.n)
n = 10
n_samples = int(args.n)
for i in range(n):
    beta = float(i)/n
    print('===','beta=', beta, '====')
    z = (1.0-beta)*z1 + beta*z2
    z = z.cuda()
    s = model.greedy_decode(z, None, None, is_gpu)
    token = s[0].detach().cpu().numpy().tolist()
    s = s[0].unsqueeze(0).long()
    smiles = tokenizer.ids2smiles(token)
    print(smiles)
    ret = model.sample_neighbor(s, n_samples, prefix=None, epsilon = epsilon, is_gpu = is_gpu)
    for b in ret:
        s = tokenizer.ids2smiles(b)
        print(s)




