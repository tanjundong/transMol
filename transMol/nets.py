import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import numpy as np
from blocks import TransEncoderLayer, TransDecoderLayer, FeedForward, GPTDecoderLayer
from blocks import ConvBottleneck, DeconvBottleneck, MeanPool
from base import Encoder, Decoder
from base import EncoderLayer, DecoderLayer


class Generator(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 vocab_size: int):

        super().__init__()

        self.proj = nn.Linear(hidden_dim, vocab_size)


    def forward(self, x):
        return self.proj(x)


class TransEncoder(Encoder):

    def __init__(self,
                 n_layers: int,
                 layer: EncoderLayer,
                 max_len: int,
                 scale=1.0):
        super().__init__()

        self.layers = nn.ModuleList([
            copy.deepcopy(layer) for _ in range(n_layers)])

        self.bottleneck = MeanPool()

        final_dim = layer.size
        self.z_means = nn.Linear(final_dim, final_dim)
        self.z_var = nn.Linear(final_dim,  final_dim)

        self.norm = nn.LayerNorm(final_dim)

        self.len_prediction = nn.Sequential(
            nn.Linear(final_dim, final_dim*2),
            nn.Linear(final_dim*2, final_dim),
            nn.ReLU(),
        )

        self.eps_scale = scale
        self.size = layer.size
        self.max_len = max_len



    def encode(self,
               x: torch.Tensor,
               mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        attens = []
        for layer in self.layers:
            #assert isinstance(layer, EncoderLayer)
            y, atten = layer.forward(x, mask)
            attens.append(atten)

        attens = torch.cat(attens, dim=0) #[nlayer, B, L, L]
        mem = self.norm(y) #[B,L,D]

        #mem = mem.permute(0, 2, 1) # [B,D,L]
        mem = self.bottleneck(mem)
        mem = mem.contiguous().view(mem.size(0), -1)
        mu, logvar = self.z_means(mem), self.z_var(mem) #[B,D]
        mem = self.reparameters(mu, logvar, self.eps_scale) #[B,D]
        pred_len = self.len_prediction(mu)

        return {
            'mem': mem,
            'mu': mu,
            'logvar': logvar,
            'pred_len': pred_len,
            'atten': attens,
        }
        #return (mu, logvar, mem, pred_len, atten)



    def reparameters(self,
                     mean: torch.Tensor,
                     logv: torch.Tensor,
                     scale: float =1.0) -> torch.Tensor:

        std = torch.exp(0.5*logv)
        eps = torch.randn_like(std) * scale
        return mean + eps*std


    def predict_property(self,
                         name: str,
                         mem: torch.Tensor) -> torch.Tensor:
        return self.len_prediction(mem)


class TransDecoder(Decoder):

    def __init__(self,
                 n_layer: int,
                 max_len: int,
                 encoder_layer : EncoderLayer,
                 layer: DecoderLayer):
        super().__init__()
        hidden_dim = layer.size
        self.encoder_layer = encoder_layer
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(n_layer)])

        self.norm = nn.LayerNorm(hidden_dim)
        encoder_dim = encoder_layer.size

        self.bridge = nn.Linear(encoder_dim, max_len*layer.size)
        self.size = layer.size
        self.max_len = max_len
        #self.linar = nn.Linear(hidden_dim, 64*9)
        #self.bottleneck = DeconvBottleneck(hidden_dim)


    def decode(self,
               x: torch.Tensor,
               mem: torch.Tensor,
               src_mask: torch.Tensor,
               tgt_mask: torch.Tensor) -> torch.Tensor:

        #mem = F.relu(self.linar(mem)) #[B, 576]

        #mem = self.bottleneck(mem)


        #print(mem.shape)
        mem = self.bridge(mem)
        mem = mem.view(-1, self.max_len, self.size)

        #mem, _ = self.encoder.forward(mem, src_mask)
        #mem = self.norm(mem)
        #print('xx, mem', x.shape, mem.shape)

        mem, _ = self.encoder_layer(mem, src_mask)
        mem = self.norm(mem)

        for layer in self.layers:
            x, _ = layer(x, mem, mem, src_mask, tgt_mask)


        return self.norm(x)


class EncoderDecoder(nn.Module):

    def __init__(self,
                 src_embedding: nn.Module,
                 tgt_embedding: nn.Module,
                 encoder: Encoder,
                 decoder: Decoder,
                 generator: Generator,
                 is_inject_latent: bool=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.is_inject_latent = is_inject_latent
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding


    def encode(self,
               x: torch.Tensor,
               src_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.src_embedding(x)
        return self.encoder.encode(x, src_mask)


    def decode(self,
               x: torch.Tensor,
               mem: torch.Tensor,
               src_mask: torch.Tensor,
               tgt_mask: torch.Tensor) -> torch.Tensor:
        x = self.tgt_embedding(x)
        return self.decoder.decode(x, mem, src_mask, tgt_mask)



    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor) -> Dict[str,torch.Tensor]:
        x = self.encode(src, src_mask)
        #mu, logvar, mem, pred_len, atten = x
        mu = x['mu']
        logvar = x['logvar']
        mem = x['mem']
        pred_len = x['pred_len']
        atten = x['atten']

        x = self.decode(tgt, mem, src_mask, tgt_mask)  #[B,L,D]
        x = self.generator(x) #[B,L,vocab_size]

        return {'logit': x,
                'mem': mem,
                'mu': mu,
                'logvar': logvar,
                'pred_len': pred_len}
        #return (x, mu, logvar, mem, pred_len)



class Embedding(nn.Module):
    pass

class PosEmbedding(Embedding):
    """PosEmbedding.
    Positional Encoding + char embedding
    """


    def __init__(self,
                 hidden_dim: int,
                 vocab_size: int,
                 max_len: int=128,
                 dropout: float = 0.1):
        """__init__.

        Parameters
        ----------
        hidden_dim : int
            hidden_dim
        vocab_size : int
            vocab_size
        max_len : int
            max_len
        dropout : float
            dropout
        """

        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.emb = nn.Embedding(vocab_size, hidden_dim)

        pe = torch.zeros(max_len, hidden_dim) #[max_len, hidden_dim]
        pos = torch.arange(0, max_len).unsqueeze(1) #[max_len, 1]
        div_ = torch.exp(torch.arange(0, hidden_dim, 2)* (-math.log(10000.0)/hidden_dim)) #[hidden_dim/2]
        pe[:, 0::2] - torch.sin(pos * div_)
        pe[:, 1::2] - torch.cos(pos * div_)
        pe = pe.unsqueeze(0) #[1, max_len, hidden_dim]

        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)


    def forward(self,
                x: torch.Tensor):
        """forward.

        Parameters
        ----------
        x : torch.Tensor [B,L]
            x token tensor, long type
        """
        y = self.emb(x) * math.sqrt(self.hidden_dim) #[B, L, D]
        self.pe.requires_grad = False
        pe = self.pe
        #print('y, pe', y.shape, pe.shape)


        z = y + pe[:, :x.size(1)]
        return self.dropout(z)



