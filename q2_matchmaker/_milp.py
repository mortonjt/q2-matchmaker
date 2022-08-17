import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl
import numpy as np
import argparse


NUM, NEUTRAL, DENOM = 0, 1, 2


class BalanceClassifier(pl.LightningModule):
    def __init__(self, input_dim, cat_dim, init_probs=None, temp=0.1,
                 learning_rate=1e-3):
        # TODO: add option to specify ILR or SLR
        super(BalanceClassifier, self).__init__()
        self.save_hyperparameters()
        if init_probs is not None:
            self.logits = nn.Parameter(init_probs)
        else:
            self.logits = nn.Parameter(torch.ones((input_dim, 3)))
        self.dist = RelaxedOneHotCategorical(logits=self.logits, temperature=temp)
        self.beta_c = nn.Parameter(torch.randn(cat_dim))  # class slope
        self.beta_b = nn.Parameter(torch.Tensor([0.01]))  # batch slope
        self.beta0 = nn.Parameter(torch.Tensor([0.01]))   # intercept
        self.lr = learning_rate

    def forward(self, trt_counts, ref_counts, trt_batch, ref_batch, hard=False):
        # sample from one hot to obtain balances
        if hard:
            x = self.dist.sample()
            balance_argmax = torch.argmax(x, axis=0).to(trt_counts.device)
            # convert to one-hot matrix
            z = torch.zeros_like(x, device=trt_counts.device)
            balance_argmax = z.scatter_(
                0, balance_argmax.unsqueeze(1), 1.)

        else:
            balance_argmax = self.dist.sample().to(trt_counts.device)
        trt_logs = torch.log(trt_counts)
        ref_logs = torch.log(ref_counts)

        trt_num = trt_logs * balance_argmax[:, NUM]
        trt_denom = trt_logs * balance_argmax[:, DENOM]
        ref_num = ref_logs * balance_argmax[:, NUM]
        ref_denom = ref_logs * balance_argmax[:, DENOM]
        trtX = torch.mean(trt_num, axis=1) - torch.mean(trt_denom, axis=1)
        refX = torch.mean(ref_num, axis=1) - torch.mean(ref_denom, axis=1)

        # classic logistic regression
        trt_ofs = trt_batch @ self.beta_c + self.beta0
        ref_ofs = ref_batch @ self.beta_c + self.beta0
        trt_logprob = self.beta_b * trtX + trt_ofs
        ref_logprob = self.beta_b * refX + ref_ofs
        res = torch.cat((trt_logprob, ref_logprob))
        return res, trtX, refX

    def logprob(self, outputs, trtX, refX):
        trt_logprob = outputs[:len(trtX)]
        ref_logprob = outputs[:len(refX)]
        N = len(trt_logprob)
        o = torch.ones(N, device=trtX.device)
        z = torch.zeros(N, device=trtX.device)
        lp = F.binary_cross_entropy_with_logits(trt_logprob, o)
        lp += F.binary_cross_entropy_with_logits(ref_logprob, z)
        return lp

    def training_step(self, batch, batch_idx):
        self.train()
        trt, ref, trt_batch, ref_batch = [x.to(self.device) for x in batch]
        outputs, trtX, refX = self.forward(
            trt, ref, trt_batch, ref_batch)
        log_prob = self.logprob(outputs, trtX, refX)
        loss = torch.sum(log_prob)
        assert torch.isnan(loss).item() is False
        current_lr = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]

        separation = torch.mean(((trtX - refX) > 0).float())
        idx = torch.randperm(len(ref))

        effect_size = torch.mean(torch.abs(trtX - refX)).float()
        tensorboard_logs = {
            'train_loss' : loss.detach().cpu().numpy(),
            'train_separation' : separation.detach().cpu().numpy(),
            'train_effect_size' : effect_size.detach().cpu().numpy(),
            'lr' : current_lr}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            trt, ref, trt_batch, ref_batch = [x.to(self.device) for x in batch]
            outputs, trtX, refX = self.forward(
                trt, ref, trt_batch, ref_batch)
            log_prob = self.logprob(outputs, trtX, refX)
            loss = torch.sum(log_prob)

            pred, trtX, refX = self.forward(trt, ref, trt_batch, ref_batch, hard=True)
            N = len(trt)

            separation = torch.mean(((trtX - refX) > 0).float())
            idx = torch.randperm(len(ref))

            effect_size = torch.mean(torch.abs(trtX - refX)).float()

            # ref_pred = self.logprob(ref[idx], ref, trt_batch, ref_batch, hard=True)
            # acc = torch.mean((pred > ref_pred).float())

            current_lr = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]
            tensorboard_logs = {
                'val_loss' : loss.detach().cpu().numpy(),
                'val_separation' : separation.detach().cpu().numpy(),
                'val_effect_size' : effect_size.detach().cpu().numpy(),
                # 'val_accuracy' : acc,
                'lr' : current_lr}
            return {'loss': loss, 'log': tensorboard_logs}


    def validation_epoch_end(self, outputs):
        metrics = [
            'val_loss',
            'val_separation',
            'val_effect_size'
            #'val_accuracy'
        ]
        tensorboard_logs = {}
        for m in metrics:
            meas = np.mean(list(map(lambda x: x['log'][m], outputs)))
            self.logger.experiment.add_scalar(m, meas, self.global_step)
            self.log(m, meas)
            tensorboard_logs[m] = meas

        # print('separation', tensorboard_logs['val_separation'],
        #       'effect_size', tensorboard_logs['val_effect_size'], '\n')
        return {'val_loss': tensorboard_logs['val_loss'],
                'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=2, T_mult=2)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser, add_help=True):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=add_help)
        parser.add_argument(
            '--init-probs', help='Path of arviz file for initialization.',
            required=False, type=str, default=None)
        parser.add_argument(
            '--learning-rate', help='Learning rate',
            required=False, type=float, default=1e-3)

        return parser


class ConditionalBalanceClassifier(BalanceClassifier):
    def __init__(self, input_dim, cat_dim, init_probs=None, temp=0.1, learning_rate=1e-3):
        # TODO: add option to specify ILR or SLR
        super(ConditionalBalanceClassifier, self).__init__(
            input_dim, cat_dim, init_probs, temp, learning_rate)
        self.save_hyperparameters()
        if init_probs is not None:
            self.logits = nn.Parameter(init_probs)
        else:
            self.logits = nn.Parameter(torch.ones((input_dim, 3)))
        self.dist = RelaxedOneHotCategorical(logits=self.logits, temperature=temp)
        self.beta_c = nn.Parameter(torch.zeros(cat_dim))  # class slope
        self.beta_b = nn.Parameter(torch.Tensor([0.01]))  # batch slope
        self.beta0 = nn.Parameter(torch.Tensor([0.01]))   # intercept
        self.lr = learning_rate

    # TODO: make sure this returns outputs rather than a loss
    def forward(self, trt_counts, ref_counts, trt_batch, ref_batch, hard=False):
        # sample from one hot to obtain balances
        if hard:
            x = self.dist.sample()
            balance_argmax = torch.argmax(x, axis=0)
            # convert to one-hot matrix
            z = torch.zeros_like(x, device=trt_counts.device)
            balance_argmax = z.scatter_(
                0, balance_argmax.unsqueeze(1), 1.)

        else:
            balance_argmax = self.dist.sample()
        trt_logs = torch.log(trt_counts)
        ref_logs = torch.log(ref_counts)

        trt_num = trt_logs * balance_argmax[:, NUM]
        trt_denom = trt_logs * balance_argmax[:, DENOM]
        ref_num = ref_logs * balance_argmax[:, NUM]
        ref_denom = ref_logs * balance_argmax[:, DENOM]
        trtX = torch.mean(trt_num, axis=1) - torch.mean(trt_denom, axis=1)
        refX = torch.mean(ref_num, axis=1) - torch.mean(ref_denom, axis=1)

        # conditional logistic regression
        trt_ofs = trt_batch @ self.beta_c + self.beta0
        ref_ofs = ref_batch @ self.beta_c + self.beta0
        trt_logprob = self.beta_b * trtX + trt_ofs
        ref_logprob = self.beta_b * refX + ref_ofs
        stack_prob = torch.stack((trt_logprob, ref_logprob))
        log_prob = trt_logprob - torch.logsumexp(stack_prob, dim=0)
        return log_prob, trtX, refX
