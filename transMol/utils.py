import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import copy


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

    mask = np.triu(np.ones(atten_shape), k=1).astype('uint18')

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
    z = F.dropout(p_atten)
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


