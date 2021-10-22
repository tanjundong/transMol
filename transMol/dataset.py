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
        self.max_len = max_len

        self.data = []

        self.init()

    def init(self):

        with open(self.path, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                ids = self.tokenizer.smiles2ids(line, self.max_len)
                np_ids = np.array(ids, dtype=np.int32)
                self.data.append(torch.from_numpy(np_ids).long())


    def __len__(self):
        return len(self.data)



    def __getitem__(self, idx):
        x = self.data[idx]

        y = x.clone()
        y = torch.roll(y, -1, dims=-1)
        #y = y[:-1] + [0]

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
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=24)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, shuffle=False, num_workers=12)




