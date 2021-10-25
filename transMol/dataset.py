import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import SmilesTokenizer
import numpy as np
import pytorch_lightning as pl

class AutoRegressionDataset(Dataset):


    def __init__(self,
                 path: str,
                 tokenizer: SmilesTokenizer,
                 is_train=True,
                 max_len=100):
        super().__init__()

        self.tokenizer = tokenizer
        self.path = path
        self.is_train = is_train
        self.max_len = max_len+1

        self.data = []
        self.smiles = []
        self.init()

    def init(self):

        tmp = []
        with open(self.path, 'r') as f:
            for i,line in enumerate(f.readlines()):
                line = line.strip('\n')
                #ids = self.tokenizer.smiles2ids(line, self.max_len)
                self.smiles.append(line)
                #if i%10000==0:
                #    print('processed lines ', i)
                #tmp.append(ids)
                #np_ids = np.array(ids, dtype=np.int32)
                #self.data.append(torch.from_numpy(np_ids).long())
        print('start tokenizing')

        for s in self.smiles:
            ids = self.tokenizer.smiles2ids(s, self.max_len)
            l = len(s)
            l = min([l, self.max_len])
            self.data.append(ids[:l+2])


    def __len__(self):
        return len(self.smiles)



    def __getitem__(self, idx):

        ids = self.data[idx]
        l = len(ids)
        x = torch.zeros(self.max_len).long()
        x[0:l] = torch.LongTensor(ids)
        #smiles = self.smiles[idx]
        #ids = self.tokenizer.smiles2ids(smiles, self.max_len)
        #x = torch.LongTensor(ids)

        y = x.clone()
        y = y[1: ]
        x = x[:-1]
        #y = torch.roll(y, -1, dims=-1)
        #y = y[:-1] + [0]
        #print(x.shape, y.shape)
        return x, y


class SmilesDataMudule(pl.LightningDataModule):

    def __init__(self,
                 tokenizer: SmilesTokenizer,
                 train_path: str,
                 val_path:str,
                 batch_size: int,
                 max_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.train_path = train_path
        self.val_path = val_path

    def setup(self, stage=None):
        self.trainset = AutoRegressionDataset(self.train_path, self.tokenizer, True, self.max_len)
        self.validset = AutoRegressionDataset(self.val_path, self.tokenizer, False, self.max_len)


    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, shuffle=False, num_workers=4)



