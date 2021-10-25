import torch
import numpy as np
from tokenizer import SmilesTokenizer



def smiles_reconstruct_accuracy(
    logit: torch.Tensor,
    gt: torch.Tensor):
    """smiles_reconstruct_accuracy.

    Parameters
    ----------
    logit : torch.Tensor [B,L,D]
        logit
    gt : torch.Tensor [B,L]
        gt
    """
    B, L, D = logit.shape

    mask = gt!=SmilesTokenizer.ID_PAD #[B,L]
    mask[:,-1] = False # don't check the last token

    pred = torch.argmax(logit, -1) #[B,L]

    gt = gt&mask
    pred = pred&mask

    #print(gt[0], pred[0], mask[0])
    gt = gt.view(-1)
    pred = pred.view(-1)

    return 1 - torch.sum(torch.abs(pred-gt)) /(B*L)

