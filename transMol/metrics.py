import torch
import torch.nn.functional as F
import numpy as np
from tokenizer import SmilesTokenizer



def smiles_reconstruct_accuracy(
    logit: torch.Tensor,
    gt: torch.Tensor,
    pad_idx=0):
    """smiles_reconstruct_accuracy.

    Parameters
    ----------
    logit : torch.Tensor
        logit
    gt : torch.Tensor
        gt
    """
    """smiles_reconstruct_accuracy.

    Parameters
    ----------
    logit : torch.Tensor [B,L,D]
        logit
    gt : torch.Tensor [B,L]
        gt
    """
    B, L, D = logit.shape

    #mask = gt!=SmilesTokenizer.ID_PAD #[B,L]
    #mask[:,-1] = False # don't check the last token

    pred = torch.argmax(logit, -1) #[B,L]

    #gt = gt&mask
    #pred = pred&mask

    #print(gt[0], pred[0], mask[0])
    gt = gt.view(-1).long()
    idx = gt!=pad_idx
    gt = gt[idx]
    pred = pred.view(-1).long()
    pred = pred[idx]
    s = gt.shape[0]
    #err = F.l1_loss(pred, gt, reduction='mean')
    err = (gt!=pred).long()
    return 1 - torch.sum(err)/(s+0.001)


