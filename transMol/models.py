import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Variable
import torch.nn.functional as F
from typing import Dict, List
import pytorch_lightning as pl
import copy
from nets import Encoder, Decoder, Embedding, EncoderDecoder, Generator
from utils import subsequent_mask, make_std_mask, is_smiles_valid
from optimizers import NoamOpt
from utils import TransformerLRScheduler

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
                 adj_predictor: nn.Module,
                 training_configs = dict()
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

        self.n_epoch = 0
        # build model
        self.model = EncoderDecoder(
            src_embedding,
            tgt_embedding,
            encoder,
            decoder,
            generator,
            True)
        self.adj_predictor = adj_predictor

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
        #length = self.predict_length(mu)
        length = 1
        #tgt = torch.ones(mu.shape[0], length+1).fill_(0).long()
        max_len = self.model.encoder.max_len
        tgt = torch.ones(mu.shape[0], self.model.encoder.max_len).fill_(3).long()


        decode_mask = subsequent_mask(max_len).long()

        if is_gpu:
            decoded = decoded.cuda()
            decode_mask = decode_mask.cuda()
            tgt = tgt.cuda()
            if prefix is not None:
                prefix = prefix.cuda()

        if prefix is not None:
            idx = prefix>0
            tgt[idx] = prefix
            tgt[:,0] = 1
        #loss_length = 2
        with torch.no_grad():
            if is_gpu:
                decode_mask = decode_mask.cuda()

                #decode_mask[:,i+1:, :] = False
                #print(decode_mask)
                #print(tgt.shape, mu.shape, src_mask.shape, decode_mask.shape)
            for i in range(max_len):
                out = self.model.decode(decoded, mu, src_mask,
                                        decode_mask)


                out = self.model.generator(out) #[B,L,vocab]
                idx = torch.argmax(out, dim=-1)
                idx = idx[:,i]
                idx = idx.unsqueeze(1)
                #print(idx.shape, decoded.shape)
                decoded = torch.cat([decoded, idx], dim=-1)


        z = decoded[:,1:]
        return z


    #def reconstruct(self, src: torch.Tensor):


    def get_prior(self) -> torch.Tensor:
        #return torch.FloatTensor([0])
        return self.prior

    def on_step(self, batch, batch_idx, is_training):
        #if is_training:
        #    self.train()
        #else:
        #    self.eval()
        #noise_src, src, y = batch #[B,L], [B,L]
        noise = batch['noise']
        src = batch['src']
        tgt = batch['tgt']
        noise = Variable(noise)
        tgt = Variable(tgt)
        pad_idx = 0
        #src = Variable(src.long())
        #tgt = Variable(src.clone())
        #tgt[:, 1:] = 1
        #src2.requires_grad = False
        src_mask = (src!=pad_idx).unsqueeze(-2)
        src_mask.requires_grad = False
        tgt_mask = make_std_mask(tgt, pad_idx)
        tgt_mask.requires_grad = False
        out = self.model.forward(noise, tgt, src_mask, tgt_mask)
        true_len = src_mask.sum(dim=-1).squeeze(-1)


        logit = out['logit']
        mem = out['mem']
        mu = out['mu']
        logvar = out['logvar']
        pred_len = out['pred_len'] #[B,D]
        #logit, mu, mem, logvar, pred_len = out

        #loss_a_mim = loss_fn.smiles_mim_loss(mu, logvar, mem, self.get_prior())
        #loss_a_mim = loss_fn.loss_mmd(mu)

        kl_weights = self.get_kl_weights()
        #loss_a_mim = loss_fn.KL_loss(mu, logvar, kl_weights)
        loss_a_mim = loss_fn.loss_mmd(mu, kl_weights)*10
        #print(logit.shape, src.shape)
        loss_bce = loss_fn.smiles_bce_loss(logit, src, pad_idx)
        loss_length = 0.0

        ret = {
            'loss_a_mim': loss_a_mim,
            'loss_bce': loss_bce,
            'loss_length': loss_length,
            'loss_adj': 0.0,
            'out': out,
            'src': src,
            'tgt': tgt,
            'noise': noise,

        }
        #print(pred_len.shape, true_len.shape)
        if self.adj_predictor is not None:
            adj = batch['adj']
            pred_adj = self.adj_predictor(mem)
            loss_adj = F.cross_entropy(pred_adj, adj, reduce='mean', ignore_index=0)
            ret['loss_adj'] = loss_adj

        return ret



    def training_step(self, batch, batch_idx):
        return self.on_step(batch, batch_idx, True)

    def training_step_end(self, batch_parts):

        loss_a_mim = batch_parts['loss_a_mim']
        loss_bce = batch_parts['loss_bce']
        loss_length = batch_parts['loss_length']
        loss_adj = batch_parts['loss_adj']

        loss_a_mim = torch.mean(loss_a_mim)
        loss_bce = torch.mean(loss_bce)
        loss_length = torch.mean(loss_length)
        loss_adj = torch.mean(loss_adj)

        self.log('train/loss_a_mim', loss_a_mim, on_step=True)
        self.log('train/loss_bce', loss_bce, on_step=True)
        self.log('train/loss_length', loss_length, on_step=True)
        self.log('train/loss_adj', loss_adj, on_step=True)


        kl_weights = self.get_kl_weights()
        total = loss_a_mim + loss_bce + loss_length + loss_adj

        self.log('train/kl_weights', kl_weights, on_step=True)
        self.log('train/loss', total, on_step=True)

        return total


    def get_kl_weights(self):

        max_epoch = self.training_configs.get('max_kl_epoch', 100)
        max_kl_weights = float(self.training_configs.get('max_kl_weights', 1.0))

        return min([max_kl_weights*float(self.n_epoch)/max_epoch, max_kl_weights])



    def validation_epoch_end(self, validation_step_outputs):
        tokenizer = SmilesTokenizer.load('./a.vocab')
        n = 100
        out = self.sample(n, tokenizer)
        n_valid = 0
        for o in out:
            is_valid = is_smiles_valid(o)
            if len(o)<3:
                is_valid = False
            if is_valid:
                n_valid +=1
        #n_valid = int(float(n_valid)/n*100.0)
        self.log("val/n_valid", n_valid)
        self.n_epoch +=1



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


        #total = loss_a_mim + loss_bce + 0*loss_length
        total = loss_a_mim + loss_bce + loss_length
        self.log('val/loss', total, on_step=True, prog_bar=True)

        tgt = batch_parts['src'] #[2xB,L]
        #print('tgt',tgt.shape)

        #tgt = torch.cat(tgt, dim=0) #[2xB, L]
        out = batch_parts['out']
        with torch.no_grad():
        #print(out)
            logit = out['logit']
            smiles_acc = metrics.smiles_reconstruct_accuracy(logit, tgt)
            self.log('val/smiles_ac', smiles_acc, on_step=True)

            tokenizer = SmilesTokenizer.load('./a.vocab')
            src = batch_parts['src']
            a = src[0].unsqueeze(0).detach()
            d = tgt[0].unsqueeze(0).detach()[0].cpu().numpy().tolist()

            #self.cuda()


            #print(a, self.tgt_embedding)
            print(' ')
            ret = self.sample_neighbor(a, 2, None)
            b = a[0].cpu().numpy().tolist()
            c = torch.argmax(logit[0], dim=-1).cpu().numpy().tolist()
            smiles = tokenizer.ids2smiles(b)
            print('origin smiles')
            print(smiles)
            print('target smiles')
            smiles = tokenizer.ids2smiles(d)
            print(smiles)
            print(tokenizer.ids2smiles(c))
            for b in ret:
                s = tokenizer.ids2smiles(b)
                valid = is_smiles_valid(s)
                print(s, valid)

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
            warmup = configs.get('warmup_steps', warmup)

        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.RNN)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.find('rnn')>-1:
                    decay.add(fpn)
                    continue
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        #no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.1},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        betas = (0.9, 0.95)
        opt = torch.optim.AdamW(optim_groups, lr=1e-4, betas=betas)
       # opt = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)


        scheduler = TransformerLRScheduler(
            optimizer=opt,
            init_lr=1e-10,
            peak_lr=0.001,
            final_lr=1e-5,
            final_lr_scale=0.05,
            warmup_steps=warmup,
            decay_steps=170000,
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step', # or 'epoch'
            'frequency': 1
        }

        return [opt], [scheduler]
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
        #pred_len = torch.argmax(pred_len, dim=-1)
        #pred_len = pred_len.item()
        ret = []
        #print(src)
        for i in range(n):
            #std = torch.exp(0.5*logvar)
            #eps = torch.rand_like(std)
            #y = std*eps
            #z = mu + y.cuda()
            z = mu + torch.randn(self.latent_dim).cuda()
            #print(z[0,0:2])
            #z = mean
            token = self.greedy_decode(z, mask, prefix)[-1]

            ret.append(token.detach().cpu().numpy().tolist())

        return ret



    def sample(self, n: int, tokenizer: SmilesTokenizer=None):


        ret = []
        mask = None
        prefix = None
        for i in range(n):
            z = torch.randn(1, self.model.encoder.size)
            z = z.cuda()
            #eps = torch.rand_like(std)
            #y = std*eps
            #z = mu + y.cuda()
            #z = mu + torch.randn(self.latent_dim).cuda()
            #print(z[0,0:2])
            token = self.greedy_decode(z, mask, prefix)[-1]
            token = token.detach().cpu().numpy().tolist()
            if tokenizer is not None:
                token = tokenizer.ids2smiles(token)

            ret.append(token)
        return ret







# factory methods


def get_model(name: str,
              configs: Dict[str, object]):

    if name=='trans':
        from nets import TransEncoder, TransDecoder, TransEncoderLayer
        from nets import TransDecoderLayer, PosEmbedding, GPTDecoderLayer
        from nets import RNNDecoder, AdjacencyPredictor


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

        #decoder_layer = TransDecoderLayer(
        #    hidden_dim,
        #    n_heads,
        #    ff_dim)

        decoder_layer = GPTDecoderLayer(
            hidden_dim,
            n_heads,
            ff_dim,
            max_len)


        decoder = TransDecoder(
            n_decode_layers,
            max_len,
            encoder_layer,
            decoder_layer,
            configs.get('decode_from_latent', False))

        decoder_type = configs.get('decoder','trans')
        if decoder_type == 'RNN':
            print('use RNN decoder')
            decoder = RNNDecoder(
                hidden_dim,
                max_len,
                n_decode_layers)


        generator = Generator(
            hidden_dim,
            vocab_size)

        predict_adj = configs.get('predict_adj', False)
        adj_predictor = None
        if predict_adj:
            adj_predictor = AdjacencyPredictor(
                hidden_dim, max_len)

        model = VAE(encoder, decoder, embedding,
                    generator,
                    #adj_predictor,
                    None,
                    training_configs=configs)
        return model



