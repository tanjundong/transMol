import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def KL_loss(mean: torch.Tensor,
              logvar: torch.Tensor,
              beta: float) -> torch.Tensor:
    """KL_loss.

    Kiebler-Lublach Divergence loss

    Parameters
    ----------
    mean : torch.Tensor [B, D]
        mean
    logvar : torch.Tensor [B, D]
        logvar
    beta : float
        beta

    Returns
    -------
    torch.Tensor [1]

    """

    a = torch.mean(
        1 + logvar - mean.pow(2) - logvar.exp())
    return -0.5*beta*a



def smiles_bce_loss(pred: torch.Tensor,
                 gt: torch.Tensor,
                 padding_idx: int) -> torch.Tensor:
    """smiles_bce_loss.

    calculate cross entropy loss for smiles sequence
    Parameters
    ----------
    pred : torch.Tensor [B,L,D]
        pred
    gt : torch.Tensor [B,L]
        gt
    padding_idx : int
        padding_idx which will be ignored in cross entropy calculation

    Returns
    -------
    torch.Tensor [1]

    """

    #gt = gt.long()[:, 1:]
    gt = gt.long()
    gt = gt.contiguous().view(-1)
    pred = pred.contiguous().view(-1, pred.size(2))

    ce = F.cross_entropy(pred, gt, ignore_index=padding_idx, reduction='mean')

    return ce


def len_bce_loss(pred_len: torch.Tensor,
                 gt_len: torch.Tensor):
    """len_bce_loss.

    smiles length prediction loss by cross entropy

    the last dimsion of pred_len should larger than the max value of gt_len
    pred_len.shape > max(gt_len)

    Parameters
    ----------
    pred_len : torch.Tensor [B, D]
        pred_len
    gt_len : torch.Tensor [B]
        gt_len
    """
    #pred_len = pred_len.contiguous().view(-1)
    gt_len = gt_len.long()
    ce = F.cross_entropy(pred_len, gt_len, reduction='mean')
    return ce


def smiles_mim_loss(mean: torch.Tensor,
                    logvar: torch.Tensor,
                    latent: torch.Tensor,
                    p_z : torch.distributions.Normal):
    """smiles_mim_loss.

    Parameters
    ----------
    mean : torch.Tensor [B,D]
        mean
    logvar : torch.Tensor [B,D]
        logvar
    latent : torch.Tensor [B,D]
        latent  z = to_var(torch.randn([batch_size, self.latent_size]))                                                                                                                        |   +encode : function
152             z = z * std + mean
    p_z : torch.distributions.Normal
        p_z
    """
    z = latent
    q_z_given_x = torch.distributions.Normal(
        loc=mean,
        scale=torch.exp(0.5*logvar)
    )
    y = q_z_given_x.log_prob(z).sum(-1)
    #print(y.sum(-1))
    y += p_z.log_prob(z).sum(-1)
    #x = torch.sum(
    #    q_z_given_x.log_prob(z).sum(-1) +
    #    p_z.log_prob(z).sum(-1))
    return -0.5*y

def compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    tiled_x = x.view(x_size,1,dim).repeat(1, y_size,1)
    tiled_y = y.view(1,y_size,dim).repeat(x_size, 1,1)

    return torch.exp(-torch.mean((tiled_x - tiled_y)**2,dim=2)/dim*1.0)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)

def loss_mmd(mean :torch.Tensor):
    true_sample = torch.randn_like(mean)
    return compute_mmd(true_sample, mean)
