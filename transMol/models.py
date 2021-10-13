from typing Dict import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import pytorch_lightning as pl
import copy
from nets import Encoder, Decoder, Embedding, EncoderDecoder, Generator

from uitls import subsequent_mask, make_std_mask
from optimizers import NoamOpt

import loss as loss_fn

class VAE(pl.LightningModule):
    """VAE.
    VAE class for ensembling, traning and molecule generation
    """


    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 embedding: Embedding,
                 generator: Generator
                 training_configs = None
                 ):

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = encoder.size()

        self.prior = torch.distributions.Normal(
            loc = self.latent_dim,
            scale= self.latent_dim)

        # embeddings
        self.src_embedding = embedding
        self.tgt_embedding = copy.deepcopy(embedding)

        self.model = EncoderDecoder(
            self.src_embedding,
            self.tgt_embedding,
            encoder,
            decoder,
            generator,
            True)

        self.training_configs = training_configs



    def encode(self,
               x: torch.Tensor,
               mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """encode.
        encode smiles into latent space

        Parameters
        ----------
        x : torch.Tensor [batch, len]
            x
        mask : torch.Tensor [batch. len]
            mask

        Returns
        -------
        Dict[str, torch.Tensor]

        """
        out = self.model.encode(x, mask)
        mem = out['mem']
        mu = out['mu']
        logvar = out['logvar']
        pred_len = out['pred_len']
        return mem, mu, logvar, pred_len, out


    def predict_length(self, mu: torch.Tensor):

        return 0

    def greedy_decode(self, mu: torch.Tensor,
                      src_mask: torch.Tensor):

        self.model.eval()
        if src_mask is None:
            pass

        decoded = torch.ones(mu.shape[0], 1).fill_(0).long()
        length = self.predict_length(mu)
        tgt = torch.ones(mu.shape[0], length+1).fill_(0).long()

        with torch.no_grad():
            for i in range(length):
                decode_mask = subsequent_mask(decoded.size(1)).long()
                out = self.model.decode(mu, src_mask,
                                        decoded,
                                        decode_mask)

                out = self.model.generator(out)
                prob = F.softmax(out[:, i, :], dim=-1)
                _, next_word = torch.max(prob, dim=1)

                tgt[:, i+1] = next_word

                next_word = next_word.unsqueeze(1)
                decoded = torch.cat([decoded, next_word], dim=1)

        z = tgt[:,1:]
        return z


    #def reconstruct(self, src: torch.Tensor):


    def get_prior(self) -> torch.Tensor:
        #return torch.FloatTensor([0])
        return self.prior

    def on_step(self, batch, batch_idx, is_training):
        if is_training:
            self.train()
        src, tgt = batch #[B,L], [B,L]
        pad_idx = -1
        src = src.long()
        tgt = tgt.long()
        src_mask = (src!=pad_idx).unsqueeze(-2)
        tgt_mask = make_std_mask(tgt, pad_idx)
        out = self.model.forward(src, tgt, src_mask, tgt_mask)
        true_len = src_mask.sum(dim=-1)

        logit = out['logit']
        mem = out['mem']
        mu = out['mu']
        logvar = out['logvar']
        pred_len = out['pred_len']

        loss_a_mim = loss_fn.smiles_mim_loss(mu, logvar, mem, self.get_prior())
        loss_bce = loss_fn.smiles_bce_loss(logit, tgt, pad_idx)
        loss_length = loss_fn.len_bce_loss(pred_len,  true_len)

        return {
            'loss_a_mim': loss_a_mim,
            'loss_bce': loss_bce,
            'loss_length': loss_length,
            'out': out,
        }

    def training_step(self, batch, batch_idx):
        self.on_step(batch, batch_idx, True)

    def training_step_end(self, batch_parts):

        loss_a_mim = batch_parts['loss_a_mim']
        loss_bce = batch_parts['loss_bce']
        loss_length = batch_parts['loss_a_length']

        loss_a_mim = torch.mean(loss_a_mim)
        loss_bce = torch.mean(loss_bce)
        loss_length = torch.mean(loss_length)

        self.log('train/loss_a_mim', loss_a_mim, on_step=True)
        self.log('train/loss_bce', loss_bce, on_step=True)
        self.log('train/loss_length', loss_length, on_step=True)


        total = loss_a_mim + loss_bce + loss_length
        self.log('train/loss', total, on_step=True)

        return total




    def validation_step(self, batch, batch_idx):
        self.on_step(batch, batch_idx, False)


    def validation_step_end(self, batch_parts):
        loss_a_mim = batch_parts['loss_a_mim']
        loss_bce = batch_parts['loss_bce']
        loss_length = batch_parts['loss_a_length']

        loss_a_mim = torch.mean(loss_a_mim)
        loss_bce = torch.mean(loss_bce)
        loss_length = torch.mean(loss_length)

        self.log('val/loss_a_mim', loss_a_mim, on_step=True)
        self.log('val/loss_bce', loss_bce, on_step=True)
        self.log('val/loss_length', loss_length, on_step=True)


        total = loss_a_mim + loss_bce + loss_length
        self.log('val/loss', total, on_step=True)


        pass

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        pass

    def configure_optimizers(self):
        dim = self.encoder.size()
        lr = 1.0
        warmup = 10000
        configs = self.training_configs
        if configs is not None:
            lr = configs.get('lr')
            warmup = configs.get('warmup_steps')

        optimizers = NoamOpt(dim, lr, warmup,
                             torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        return [optimizers]






# factory methods


def get_model(name: str
              configs: Dict[str, object]):

    if name=='trans':
        from nets import TransEncoder, TransDecoder, TransDecoderLayer, TransDecoderLayer, PosEmbedding
        hidden_dim = configs.get('hidden_dim', 128)
        ff_dim = configs.get('ff_dim', 128)
        max_len = configs.get('max_len', 128)
        vocab_size = configs.get('vocab_size', 100)
        n_heads = configs.get('n_heads', 8)
        n_encode_layers = configs.get('n_encode_layers', 6)
        n_decode_layers = configs.get('n_decode_layers', 6)

        encoder_layer = TransEncoderLayer(
            hidden_dim,
            n_heads,
            ff_dim)
        encoder = TransEncoder(
            n_encode_layers,
            encoder_layer,
            max_len)

        embedding = PosEmbedding(hidden_dim, vocab_size, max_len)

        decoder_layer = DecoderLayer(
            hidden_dim,
            n_heads,
            ff_dim)
        decoder = TransDecoder(
            n_decode_layers,
            encoder_layer,
            decoder_layer)


        generator = Generator(
            hidden_dim,
            vocab_size)


        model = VAE(encoder, decoder, embedding, generator)
        return model











