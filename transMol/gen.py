import torch
from models import get_model, VAE

from dataset import SmilesDataMudule
import pytorch_lightning as pl
from tokenizer import SmilesTokenizer
import sys
from configs import configs

tokenizer = SmilesTokenizer.load('./a.vocab')

model = get_model('trans', configs)
assert isinstance(model, pl.LightningModule)

model.load('./transMol-transMol/a.ckpt')

#smiles = 'CC(=O)NC1CCC2(C)C(CCC3(C)C2C(=O)C=C2C4C(C)C(C)CCC4(C)CCC23C)C1(C)C(=O)O'
#smiles = sys.argv[-1]
#token = tokenizer.smiles2ids(smiles, configs['max_len'])
model.cuda()
#a = torch.LongTensor(token).unsqueeze(0).cuda()
#a = torch.roll(a, -1, -1)

#ret = model.sample_neighbor(a, 10)
ret = model.sample(10)
for b in ret:
    s = tokenizer.ids2smiles(b)
    print(s)




