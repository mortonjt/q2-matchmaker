import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical
import pytorch_lightning as pl


NUM, NEUTRAL, DENOM = 0, 1, 2


class ConditionalBalanceClassifier(pl.LightningModule):
    def __init__(self, input_dim, init_probs=None, temp=0.1):
        
        if init_probs is not None:
            self.logits = nn.Parameter(init_probs)
        else:
            self.logits = nn.Parameter(torch.ones((input_dim, 3)))
        self.dist = RelaxedOneHotCategorical(logits=logits, temperature=temp)
        self.beta = nn.Parameter(0.01)

    def forward(self, trt_counts, ref_counts, batch_id):
        # sample from one hot to obtain balances
        balance_argmax = F.softmax(self.dist.sample())
        trt_logs = torch.log(trt_counts)
        ref_logs = torch.log(ref_counts)
        trt_parts = trt_logs * balance_argmax
        ref_parts = ref_logs * balance_argmax
        trt_X = trt_parts[NUM].mean() - trt_parts[DENOM].mean()
        ref_X = ref_parts[NUM].mean() - ref_parts[DENOM].mean()
        # conditional logistic regression
        loglike = self.beta * trt_X - torch.logsumexp(self.beta * trtX, self.beta * refX)        
        return loglike
    
