import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import SmilesTokenizer
import numpy as np
import pytorch_lightning as pl
from rdkit import Chem

class AutoRegressionDataset(Dataset):


    def __init__(self,
                 path: str,
                 tokenizer: SmilesTokenizer,
                 is_train=True,
                 is_denoising = True,
                 max_len=100):
        super().__init__()

        self.tokenizer = tokenizer
        self.path = path
        self.is_train = is_train
        self.max_len = max_len
        self.is_denoising = is_denoising

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

        '''
        for s in self.smiles:
            ids = self.tokenizer.smiles2ids(s, self.max_len)
            l = len(s)
            l = min([l, self.max_len])
            self.data.append(ids[:l+2])
        '''

    def __len__(self):
        return len(self.smiles)

    def get_noise_mask(self, l):
        p = 0.15
        ret = torch.FloatTensor(self.max_len).uniform_()<p
        ret[l:] = False
        return ret


    def __getitem__(self, idx):

        #ids = self.data[idx]
        #l = len(ids)
        #x = torch.zeros(self.max_len).long()
        #x[0:l] = torch.LongTensor(ids)
        #smiles = self.smiles[idx]
        #ids = self.tokenizer.smiles2ids(smiles, self.max_len)
        #x = torch.LongTensor(ids)

        smiles = self.smiles[idx]

        smiles = self.permute_smiles(smiles)
        if smiles is None:
            smiles = self.smiles[idx]

        ids = self.tokenizer.smiles2ids(smiles, self.max_len)
        y = [self.tokenizer.ID_SOS] + ids[:-1]
        l = len(ids)
        x = torch.LongTensor(ids)
        y = torch.LongTensor(y)

        noise_x = x.clone()
        if self.is_denoising:
            m = self.get_noise_mask(l)
            noise_x = x.masked_fill_(m, SmilesTokenizer.ID_MASK)
            #y = torch.cat([self.tokenizer.ID_SOS], noise_x[:-1])
            y = torch.Tensor([self.tokenizer.ID_SOS]) + noise_x[:-1]
            y = y.long()
            #idx = x!=SmilesTokenizer.ID_MASK
            #tmp = x[idx]
            #x[0:tmp.shape[-1]] = tmp


        #return noise_x, x, y
        return {
            'noise': noise_x,
            'src': x,
            'tgt': y
        }

    def permute_smiles(self, smiles_str: str, seed: int = None):
        """
        Permute the input smiles.

        Args:
          smiles_str: The smiles input

        Returns:
          The standardised permuted smiles.
        """
        if seed is not None:
            np.random.seed(seed)

        try:
            mol = Chem.MolFromSmiles(smiles_str, sanitize=False)
        except Exception as e:
            logging.warning(f'Chem.MolFromSmiles failed smiles="{smiles_str}" error={e}')
            return None

        if mol is None:
            # invalid?
            return None
        ans = list(range(mol.GetNumAtoms()))
        np.random.shuffle(ans)

        # re-order the molecule
        smiles = Chem.MolToSmiles(Chem.RenumberAtoms(mol, ans), canonical=False)

        # standardise and return
        #return self.standardise(smiles)
        return smiles

    def standardise(self, smiles: str, canonicalise:bool = None) -> str:
        """
        Standardise a SMILES string if valid (canonical + kekulized)

        Args:
            smiles: SMILES string
            canonicalise: optional flag to override `self.canonicalise`

        Returns: standard version the SMILES if valid, None otherwise

        """
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
        except Exception as e:
            # invalid?
            logging.warning(f'Chem.MolFromSmiles failed smiles="{smiles}" error={e}')
            return None

        if mol is None:
            # invalid?
            logging.warning(f'Chem.MolFromSmiles failed smiles="{smiles}"')
            return None

        flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_CLEANUP
        Chem.SanitizeMol(mol, flags, catchErrors=True)

        if canonicalise:
            # bug where permuted smiles are not canonicalised to the same form. This is fixed by round tripping SMILES
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            if mol is None:
                logging.warning(f'Chem.MolFromSmiles failed after sanitization smiles="{smiles}"')
                return None

        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
            smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=canonicalise)
        except (ValueError, RuntimeError):
            logging.warning(f'SMILES failed Kekulization! {smiles}')
            return None

        return smiles


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

        self.trainset = AutoRegressionDataset(self.train_path,
                                              self.tokenizer, True, False, self.max_len)
        self.validset = AutoRegressionDataset(self.val_path,
                                              self.tokenizer, True, False, self.max_len)



    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, shuffle=False, num_workers=10)




