import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from utils.sgd_hat import SGD_hat as SGD
from apprs.basemodel import BaseModel
from collections import Counter
from copy import deepcopy
from utils.utils import *
from torch.utils.data import DataLoader
from collections import Counter

device = "cuda" if torch.cuda.is_available() else "cpu"

class HAT(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        
    def update_s(self, b, B):
        """ b: current batch, B: total num batch """
        s = (self.args.smax - 1 / self.args.smax) * b / B + 1 / self.args.smax
        return s

    def compensation(self, model, thres_cosh=50, s=1):
        """ Equation before Eq. (4) in the paper """
        for n, p in model.named_parameters():
            if 'ec' in n:
                if p.grad is not None:
                    num = torch.cosh(torch.clamp(s * p.data, -thres_cosh, thres_cosh)) + 1
                    den = torch.cosh(p.data) + 1
                    p.grad *= self.args.smax / s * num / den

    def compensation_clamp(self, model, thres_emb=6):
        # Constrain embeddings
        for n, p in model.named_parameters():
            if 'ec' in n:
                if p.grad is not None:
                    p.data.copy_(torch.clamp(p.data, -thres_emb, thres_emb))

    def modify_grad(self, model, mask_back):
        """ 
            Zero-out gradients if both masks are 1. Eq. (2) in the paper
            Gradients of convolutions
        """
        for n, p in model.named_parameters():
            if n in mask_back:
                p.grad *= mask_back[n]

    def hat_reg(self, p_mask, masks):
        """ masks and self.p_mask must have values in the same order """
        reg, count = 0., 0.
        if p_mask is not None:
            for m, mp in zip(masks, p_mask.values()):
                aux = 1. - mp#.to(device)
                reg += (m * aux).sum()
                count += aux.sum()
            reg /= count
            return self.args.lamb1 * reg
        else:
            for m in masks:
                reg += m.sum()
                count += np.prod(m.size()).item()
            reg /= count
            return self.args.lamb0 * reg