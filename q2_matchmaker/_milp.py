import torch
import torch.nn as nn
from torch.distributions import RelaxedOneHotCategorical
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl
import numpy as np
import argparse


NUM, NEUTRAL, DENOM = 0, 1, 2


class ConditionalBalanceClassifier(pl.LightningModule):
    def __init__(self, input_dim, cat_dim, init_probs=None, temp=0.1, learning_rate=1e-3):
        # TODO: add option to specify ILR or SLR
        super(ConditionalBalanceClassifier, self).__init__()
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

    def forward(self, trt_counts, ref_counts, trt_batch, ref_batch, hard=False):
        # sample from one hot to obtain balances
        if hard:
            x = self.dist.sample()
            balance_argmax = torch.argmax(x, axis=0)
            # convert to one-hot matrix
            balance_argmax = torch.zeros_like(x).scatter_(
                0, balance_argmax.unsqueeze(1), 1.)

        else:
            balance_argmax = self.dist.sample()
        trt_logs = torch.log(trt_counts)
        ref_logs = torch.log(ref_counts)

        trt_num = trt_logs * balance_argmax[:, NUM]
        trt_denom = trt_logs * balance_argmax[:, DENOM]
        ref_num = ref_logs * balance_argmax[:, NUM]
        ref_denom = ref_logs * balance_argmax[:, DENOM]
        trtX = torch.mean(trt_num) - torch.mean(trt_denom)
        refX = torch.mean(ref_num) - torch.mean(ref_denom)

        # conditional logistic regression
        trt_ofs = trt_batch @ self.beta_c + self.beta0
        ref_ofs = ref_batch @ self.beta_c + self.beta0
        trt_logprob = self.beta_b * trtX + trt_ofs
        ref_logprob = self.beta_b * refX + ref_ofs
        stack_prob = torch.stack((trt_logprob, ref_logprob))
        log_prob = trt_logprob - torch.logsumexp(stack_prob, dim=0)
        return log_prob

    def training_step(self, batch, batch_idx):
        trt, ref, trt_batch, ref_batch = [x.to(self.device) for x in batch]
        log_prob = self.forward(trt, ref, trt_batch, ref_batch)
        loss = - torch.sum(log_prob)
        current_lr = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]
        tensorboard_logs = {'train_loss' : loss, 'lr' : current_lr}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            trt, ref, trt_batch, ref_batch = [x.to(self.device) for x in batch]
            log_prob = self.forward(trt, ref, trt_batch, ref_batch)
            loss = - torch.sum(log_prob)

            pred = self.forward(trt, ref, trt_batch, ref_batch, hard=True)
            separation = torch.mean((pred > 0).float())
            idx = torch.randperm(len(ref))
            ref_pred = self.forward(ref[idx], ref, trt_batch, ref_batch, hard=True)
            acc = torch.mean((pred > ref_pred).float())

            current_lr = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]
            tensorboard_logs = {'val_loss' : loss,
                                'val_separation' : separation,
                                'val_accuracy' : acc,
                                'lr' : current_lr}
            return {'loss': loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        metrics = ['val_loss',
                   'val_separation',
                   'val_accuracy']
        tensorboard_logs = {}
        for m in metrics:
            meas = np.mean(list(map(lambda x: x['log'][m], outputs)))
            self.logger.experiment.add_scalar(m, meas, self.global_step)
            self.log(m, meas)
            tensorboard_logs[m] = meas
        return {'val_loss': tensorboard_logs['val_loss'],
                'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams['learning_rate'],
            weight_decay=0)
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
        parser.add_argument(
            '--output-dir', help='Path to save all logging files and checkpoints',
            required=True, type=str, default=None)
        return parser
