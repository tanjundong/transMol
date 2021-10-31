import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Variable
import torch.nn.functional as F
from typing import Dict, List
import pytorch_lightning as pl
import copy
from nets import Encoder, Decoder, Embedding, EncoderDecoder, Generator
from utils import subsequent_mask, make_std_mask
from optimizers import NoamOpt

import loss as loss_fn
import metrics
from tokenizer import SmilesTokenizer

class VAE(pl.LightningModule):
    """VAE.
    VAE class for ensembling, traning and molecule generation
    """


    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 embedding: Embedding,
                 generator: Generator,
                 training_configs = None
                 ):
        """__init__.

        Parameters
        ----------
        encoder : Encoder
            encoder
        decoder : Decoder
            decoder
        embedding : Embedding
            embedding
        generator : Generator
            generator
        training_configs :
            training_configs
        """

        super().__init__()

        #self.encoder = encoder
        #self.decoder = decoder
        self.latent_dim = encoder.size
        #self.hidden_size = encoder.size

        self.register_buffer("mean", Variable(torch.zeros(self.latent_dim)))
        self.register_buffer("var", Variable(torch.ones(self.latent_dim)))
        self.mean.requires_grad = False
        self.var.requires_grad = False
        #self.prior = torch.distributions.Normal(
        #    loc = torch.zeros(self.latent_dim),
        #    scale= torch.ones(self.latent_dim),
        #)
        #self.init_prior()

        #self.register_buffer('prior', prior)

        # embeddings
        src_embedding = embedding
        tgt_embedding = copy.deepcopy(embedding)


        # build model
        self.model = EncoderDecoder(
            src_embedding,
            tgt_embedding,
            encoder,
            decoder,
            generator,
            True)

        self.training_configs = training_configs


    @property
    def prior(self):
        return torch.distributions.Normal(
            loc = self.mean,
            scale= self.var,
        )


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
        return  mu, logvar, mem, pred_len, out


    def predict_length(self, mu: torch.Tensor):

        #return 0
        encoder = self.model.encoder
        y = encoder.len_prediction(mu)
        return torch.argmax(y).detach().cpu().item()


    def greedy_decode(self, mu: torch.Tensor,
                      src_mask: torch.Tensor,
                      prefix : torch.Tensor=None,
                      is_gpu=True):

        self.model.eval()


        decoded = torch.ones(mu.shape[0], 1).fill_(1).long()
        if src_mask is None:
            pass
        length = self.predict_length(mu)
        #tgt = torch.ones(mu.shape[0], length+1).fill_(0).long()
        max_len = self.model.encoder.max_len
        tgt = torch.ones(mu.shape[0], self.model.encoder.max_len).fill_(3).long()


        decode_mask = subsequent_mask(max_len).long()

        if is_gpu:
            decoded = decoded.cuda()
            decode_mask = decode_mask.cuda()
            tgt = tgt.cuda()
            prefix = prefix.cuda()

        if prefix is not None:
            idx = prefix>0
            tgt[idx] = prefix
            tgt[:,0] = 1

        with torch.no_grad():
            for i in range(length):
                decode_mask = subsequent_mask(decoded.size(1)).long()
                if is_gpu:
                    decode_mask = decode_mask.cuda()

                #decode_mask[:,i+1:, :] = False
                #print(decode_mask)
                #print(tgt.shape, mu.shape, src_mask.shape, decode_mask.shape)
                out = self.model.decode(decoded, mu, src_mask,
                                        decode_mask)


                out = self.model.generator(out) #[B,L,vocab]
                #prob = F.softmax(out[:, i, :], dim=-1)
                #_, next_word = torch.max(prob, dim=1)
                idx = torch.argmax(out, dim=-1)
                next_word = idx[:, i]
                tgt[:, i+1] = next_word

                next_word = next_word.unsqueeze(1)
                decoded = torch.cat([decoded, next_word], dim=1)
                decoded = decoded.long()
                if i>=max_len-2:
                    break

        z = tgt[:,1:]
        return z


    #def reconstruct(self, src: torch.Tensor):


    def get_prior(self) -> torch.Tensor:
        #return torch.FloatTensor([0])
        return self.prior

    def on_step(self, batch, batch_idx, is_training):
        if is_training:
            self.train()
        src, y = batch #[B,L], [B,L]
        pad_idx = 0
        src = Variable(src.long())
        tgt = Variable(src.clone())
        #src2.requires_grad = False
        src_mask = (src!=pad_idx).unsqueeze(-2)
        src_mask.requires_grad = False
        tgt_mask = make_std_mask(tgt, pad_idx)
        tgt_mask.requires_grad = False
        out = self.model.forward(src, y, src_mask, tgt_mask)
        true_len = src_mask.sum(dim=-1).squeeze(-1)


        logit = out['logit']
        mem = out['mem']
        mu = out['mu']
        logvar = out['logvar']
        pred_len = out['pred_len'] #[B,D]
        #logit, mu, mem, logvar, pred_len = out

        #loss_a_mim = loss_fn.smiles_mim_loss(mu, logvar, mem, self.get_prior())
        #loss_a_mim = loss_fn.loss_mmd(mu)
        loss_a_mim = loss_fn.KL_loss(mu, logvar, 0.5)
        loss_bce = loss_fn.smiles_bce_loss(logit, tgt, pad_idx)
        #print(pred_len.shape, true_len.shape)

        loss_length = loss_fn.len_bce_loss(pred_len,  true_len)
        #loss_length = 0.0
        return {
            'loss_a_mim': loss_a_mim,
            'loss_bce': loss_bce,
            'loss_length': loss_length,
            'out': out,
            'src': src,
            'tgt': tgt,
        }



    def training_step(self, batch, batch_idx):
        return self.on_step(batch, batch_idx, True)

    def training_step_end(self, batch_parts):

        loss_a_mim = batch_parts['loss_a_mim']
        loss_bce = batch_parts['loss_bce']
        loss_length = batch_parts['loss_length']

        loss_a_mim = torch.mean(loss_a_mim)
        loss_bce = torch.mean(loss_bce)
        loss_length = torch.mean(loss_length)

        self.log('train/loss_a_mim', loss_a_mim, on_step=True)
        self.log('train/loss_bce', loss_bce, on_step=True)
        self.log('train/loss_length', loss_length, on_step=True)


        total = loss_a_mim + loss_bce + loss_length

        self.log('train/loss', total, on_step=True)

        return total


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pad_idx = 0
        src = x.long()
        tgt = src
        src_mask = (src!=pad_idx).unsqueeze(-2)
        tgt_mask = make_std_mask(tgt, pad_idx)
        out = self.model.forward(src, tgt, src_mask, tgt_mask)
        return out



    def validation_step(self, batch, batch_idx):

        #token = tokenizer.smiles2ids(smiles, self.model.encoder.max_len)
        #a = torch.LongTensor(token).unsqueeze(0).cuda()
        #src, tgt = batch


        return self.on_step(batch, batch_idx, False)


    def validation_step_end(self, batch_parts):
        loss_bce = batch_parts['loss_bce']
        loss_length = batch_parts['loss_length']
        loss_a_mim = batch_parts['loss_a_mim']
        #print(loss_bce)
        loss_a_mim = torch.mean(loss_a_mim)
        loss_bce = torch.mean(loss_bce)
        loss_length = torch.mean(loss_length)


        self.log('val/loss_a_mim', loss_a_mim, on_step=True)
        self.log('val/loss_bce', loss_bce, on_step=True)
        self.log('val/loss_length', loss_length, on_step=True)


        total = loss_a_mim + loss_bce + loss_length
        self.log('val/loss', total, on_step=True)

        tgt = batch_parts['tgt'] #[2xB,L]
        #print('tgt',tgt.shape)

        #tgt = torch.cat(tgt, dim=0) #[2xB, L]
        out = batch_parts['out']
        #print(out)
        logit = out['logit']
        smiles_acc = metrics.smiles_reconstruct_accuracy(logit, tgt)
        self.log('val/smiles_ac', smiles_acc)

        tokenizer = SmilesTokenizer.load('./a.vocab')
        src = batch_parts['src']
        a = src[0].unsqueeze(0).detach()

        #self.cuda()


        #print(a, self.tgt_embedding)
        print('*'*20)
        c = torch.ones_like(a)
        c[:, 0:3] =  a[:, 0:3]
        ret = self.sample_neighbor(a, 2, None)
        b = a[0].cpu().numpy().tolist()
        smiles = tokenizer.ids2smiles(b)
        print('origin smiles', smiles)
        for b in ret:
            s = tokenizer.ids2smiles(b)
            print(s)

        print('='*20)


        return total



    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        pass

    def configure_optimizers(self):
        dim = self.latent_dim
        lr = 1.0
        warmup = 10000
        configs = self.training_configs
        if configs is not None:
            lr = configs.get('lr')
            warmup = configs.get('warmup_steps')

        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)


        return opt
        #optimizers = NoamOpt(dim, lr, warmup,
        #                     torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


        #return optimizers


    def load(self, path):
        ckpt = torch.load(path)
        state_dict = ckpt['state_dict']
        self.load_state_dict(state_dict)


    def sample_neighbor(self, src: torch.Tensor, n: int, prefix = None):
        mask = (src!=0).unsqueeze(-2)  #[B,1,L]
        mu, logvar, mean, pred_len, out = self.encode(src, mask)
        pred_len = torch.argmax(pred_len, dim=-1)
        pred_len = pred_len.item()
        ret = []
        #print(src)
        for i in range(n):
            #z = mu + torch.randn(self.latent_dim).cuda()
            #print(z[0,0:2])
            z = mean
            token = self.greedy_decode(z, mask, prefix)[-1]

            ret.append(token.detach().cpu().numpy().tolist())

        return ret








# factory methods


def get_model(name: str,
              configs: Dict[str, object]):

    if name=='trans':
        from nets import TransEncoder, TransDecoder, TransEncoderLayer, TransDecoderLayer, PosEmbedding
        hidden_dim = configs.get('hidden_dim', 128)
        ff_dim = configs.get('ff_dim', 128)
        max_len = configs.get('max_len', 100)
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

        decoder_layer = TransDecoderLayer(
            hidden_dim,
            n_heads,
            ff_dim)

        decoder = TransDecoder(
            n_decode_layers,
            max_len,
            encoder_layer,
            decoder_layer)


        generator = Generator(
            hidden_dim,
            vocab_size)

        model = VAE(encoder, decoder, embedding, generator)
        return model



