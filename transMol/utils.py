import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Optional
from typing import Dict, List
import copy
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def subsequent_mask(size: int) -> torch.Tensor:
    """subsequent_mask.
    Parameters
    ----------
    size : int
        size

    Returns
    -------
    torch.Tensor [1, size, size]

    """

    atten_shape = (1, size, size)

    mask = np.triu(np.ones(atten_shape), k=1).astype('uint8')

    return torch.from_numpy(mask) == 0

def attention(q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              mask: torch.Tensor,
              dropout: float=0.1) ->[torch.Tensor, torch.Tensor]:
    """attention function.

    Parameters
    ----------
    q : torch.Tensor [B, H, L, K]
        q
    k : torch.Tensor [B, H, L, K]
        k
    v : torch.Tensor [B, H, L, K]
        v
    mask : torch.Tensor [B, 1, L, L]
        mask
    dropout : float
        dropout

    Returns
    -------
    [torch.Tensor, torch.Tensor]

    """

    d_k = q.shape[-1]

    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) # [B, H, L, L]

    if mask is not None:
        score = score.masked_fill(mask==0, -1e9)

    p_atten = F.softmax(score, dim=-1) #[B, H, L, L]
    z = F.dropout(p_atten, dropout)
    y = torch.matmul(z, v) # [B, H, L, K]

    return y, p_atten


def make_std_mask(tgt:torch.Tensor,
                  pad: int) -> torch.Tensor:
    """make_std_mask.

    Parameters
    ----------
    tgt : torch.Tensor [B,L]
        tgt
    pad : int
        pad

    Returns
    -------
    torch.Tensor

    """

    tgt_mask = (tgt!=pad).unsqueeze(-2) # [B,1,L]

    m = subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data) #[1,L,L]
    return tgt_mask & m


class LearningRateScheduler(torch.optim.lr_scheduler._LRScheduler):
    r"""
    Provides inteface of learning rate scheduler.
    Note:
        Do not use this class directly, use one of the sub classes.
    """

    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr

    def step(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']



class TransformerLRScheduler(LearningRateScheduler):
    r"""
    Transformer Learning Rate Scheduler proposed in "Attention Is All You Need"
    Args:
        optimizer (Optimizer): Optimizer.
        init_lr (float): Initial learning rate.
        peak_lr (float): Maximum learning rate.
        final_lr (float): Final learning rate.
        final_lr_scale (float): Final learning rate scale
        warmup_steps (int): Warmup the learning rate linearly for the first N updates
        decay_steps (int): Steps in decay stages
    """
    def __init__(
            self,
            optimizer: Optimizer,
            init_lr: float,
            peak_lr: float,
            final_lr: float,
            final_lr_scale: float,
            warmup_steps: int,
            decay_steps: int,
    ) -> None:
        assert isinstance(warmup_steps, int), "warmup_steps should be inteager type"
        assert isinstance(decay_steps, int), "total_steps should be inteager type"

        super(TransformerLRScheduler, self).__init__(optimizer, init_lr)
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

        self.warmup_rate = self.peak_lr / self.warmup_steps
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        self.init_lr = init_lr
        self.update_steps = 0

    def _decide_stage(self):
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps

        if self.warmup_steps <= self.update_steps < self.warmup_steps + self.decay_steps:
            return 1, self.update_steps - self.warmup_steps

        return 2, None

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        self.update_steps += 1
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.update_steps * self.warmup_rate
        elif stage == 1:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 2:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.optimizer, self.lr)

        return self.lr

def is_smiles_valid(smiles: str):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is not None:
        return True
    else:
        return False
