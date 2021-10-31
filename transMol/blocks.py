import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import copy

from utils import attention

_BOTTLENECK_KERNEL_SIZE = 9
_BOTTLENECK_CHANNEL = 64
_DROPOUT = 0.2
class MultiheadAttention(nn.Module):


    def __init__(self,
                 hidden_dim: int,
                 n_heads: int,
                 dropout: float = _DROPOUT):

        super().__init__()
        assert hidden_dim % n_heads == 0

        self.d_k = hidden_dim // n_heads

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout

    def reshape(self,
                x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.view(B, -1, self.n_heads, self.d_k).transpose(1, 2) # [B, H, L, K]
        return x

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        """forward.

        Parameters
        ----------
        q : torch.Tensor [B, L, D]
            q
        k : torch.Tensor [B, L, D]
            k
        v : torch.Tensor [B, L, D]
            v
        mask : torch.Tensor [B, L]
            mask

        Returns
        -------
        [torch.Tensor, torch.Tensor]

        atten_result: [B,L,D]
        atten_map: [B, H, L, L]
        """


        B, L, D = q.shape
        K = self.d_k
        query = self.q(q)
        key = self.k(k)
        value = self.v(v)

        query = self.reshape(query)
        key = self.reshape(key)
        value = self.reshape(value)
        if mask is not None:
            if len(mask.shape)!=4:
                mask = mask.unsqueeze(1) #[B, 1, L, L]

        z, atten = attention(query, key, value, mask, self.dropout)
        #print(z.shape, atten.shape)

        z = z.transpose(1,2) # [B, L, H, K]
        z = z.contiguous().view(B, L, D)
        z = self.out(z)
        return z, atten





class FeedForward(nn.Module):
    """FeedForward.

    Postitionwise Feed Forward layer

    """


    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 dropout: float= 0.5):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                 #nn.ReLU(inplace=True),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, in_dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class SkipConnection(nn.Module):
    """SkipConnection.

    y = x + dropout(sub_layer(norm(x)))
    """


    def __init__(self,
                 dim: int,
                 dropout: float=_DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor):
        z = x + self.dropout(self.norm(y))
        #z = self.dropout(self.norm(y))
        return z


class TransEncoderLayer(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 n_heads: int,
                 ff_dim: int,
                 dropout: float=_DROPOUT):
        super().__init__()

        assert hidden_dim % n_heads == 0
        self.self_attention_1 = MultiheadAttention(hidden_dim, n_heads, dropout)
        self.residule_1 = SkipConnection(hidden_dim, dropout)

        #self.self_attention_2 = nn.MultiheadAttention(hidden_dim, n_heads, dropout)
        self.residule_2 = SkipConnection(hidden_dim, dropout)

        self.ff = FeedForward(hidden_dim, ff_dim, dropout)
        self.size = hidden_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.norm = nn.LayerNorm(hidden_dim)



    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        """forward.

        Parameters
        ----------
        x : torch.Tensor [B, L, D]
            x
        mask : torch.Tensor [B, L, L]
            mask

        Returns
        -------
        [torch.Tensor, torch.Tensor]
        [B,L,D] [B,H,L,L]

        """

        B, L, D = x.shape

        y, atten = self.self_attention_1(self.norm(x), self.norm(x), self.norm(x), mask)
        #y = x + self.norm(y)

        #y = self.residule_1(x, y)

        z = self.ff(self.norm(y))
        #z = z + y
        #z = self.residule_2(y, z)

        #w = self.ff(z)
        return z, atten



class TransDecoderLayer(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 n_heads: int,
                 ff_dim: int,
                 dropout: float = _DROPOUT):

        super().__init__()
        self.src_atten = MultiheadAttention(hidden_dim, n_heads, dropout)
        self.self_atten = MultiheadAttention(hidden_dim, n_heads, dropout)
        self.ff = FeedForward(hidden_dim, ff_dim, dropout)
        self.residule_1 = SkipConnection(hidden_dim, dropout)
        self.residule_2 = SkipConnection(hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        self.size = hidden_dim
        self.dropout = nn.Dropout(dropout)


    def forward(self,
                x: torch.Tensor,
                mem_key: torch.Tensor,
                mem_val: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor) -> [torch.Tensor, torch.Tensor]:

        x = self.norm(x)
        y, _ = self.self_atten(x, x, x, tgt_mask)
        #y = x + self.dropout(y)

        z, src_att = self.src_atten(self.norm(y), mem_key, mem_val, src_mask)
        #z = y + self.dropout(z)

        w = self.ff(self.norm(z))

        return w, src_att




        #y = self.residule_1(x, y)

        #z, tgt_att = self.tgt_atten(y, mem_key, mem_val, src_mask)
        #z = self.ff(z)
        #z = self.residule_2(y, z)


        #return z, src_att



class ConvBottleneck(nn.Module):
    """
    Set of convolutional layers to reduce memory matrix to single
    latent vector
    """
    def __init__(self, size):
        super().__init__()
        conv_layers = []
        in_d = size
        first = True
        for i in range(3):
            out_d = int((in_d - _BOTTLENECK_CHANNEL) // 2 + _BOTTLENECK_CHANNEL)
            if first:
                kernel_size = _BOTTLENECK_KERNEL_SIZE
                first = False
            else:
                kernel_size = _BOTTLENECK_KERNEL_SIZE
            if i == 2:
                out_d = _BOTTLENECK_KERNEL_SIZE
            pad = kernel_size//2+1
            conv_layers.append(nn.Sequential(nn.Conv1d(in_d, out_d, kernel_size, padding=pad), nn.MaxPool1d(2)))


            in_d = out_d
        #print(len(conv_layers))

        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x):
        for i, conv in enumerate(self.conv_layers):
            print(i, x.shape)
            x = F.relu(conv(x))
        return x

    def get_final_dim(self, max_len):
        out_dim = max_len
        test = torch.ones(1, 128, max_len)
        out = self.forward(test)
        return out.shape[-1]


class DeconvBottleneck(nn.Module):
    """
    Set of deconvolutional layers to reshape latent vector
    back into memory matrix
    """
    def __init__(self, size):
        super().__init__()
        deconv_layers = []
        in_d = _BOTTLENECK_CHANNEL
        for i in range(3):
            out_d = (size - in_d) // 4 + in_d
            stride = 4 - i
            kernel_size = _BOTTLENECK_KERNEL_SIZE + 2
            if i == 2:
                out_d = size
                stride = 1
            deconv_layers.append(nn.Sequential(nn.ConvTranspose1d(in_d, out_d, kernel_size,
                                                                  stride=stride, padding=2)))
            in_d = out_d
        self.deconv_layers = nn.ModuleList(*deconv_layers)

    def forward(self, x):
        for deconv in self.deconv_layers:
            x = F.relu(deconv(x))
        return x



class MeanPool(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self,
                x: torch.Tensor):

        #B,L,D = x.shape
        # to skip start token
        y = torch.mean(x[:,1:,:], dim=1, keepdim=False)


        return y
