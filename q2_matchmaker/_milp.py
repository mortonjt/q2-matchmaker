import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl


NUM, NEUTRAL, DENOM = 0, 1, 2


class ConditionalBalanceClassifier(pl.LightningModule):
    def __init__(self, input_dim, init_probs=None, temp=0.1, lr=1e-3):
        self.save_hyperparameters()
        if init_probs is not None:
            self.logits = nn.Parameter(init_probs)
        else:
            self.logits = nn.Parameter(torch.ones((input_dim, 3)))
        self.dist = RelaxedOneHotCategorical(logits=self.logits, temperature=temp)
        self.beta_c = nn.Parameter([0.01])  # class slope
        self.beta_b = nn.Parameter([0.01])  # batch slope
        self.beta0 = nn.Parameter([0.01])   # intercept
        

    def forward(self, trt_counts, ref_counts, batch_id, hard=False):
        # sample from one hot to obtain balances
        if hard:
            balance_argmax = torch.argmax(self.dist.sample(), axis=0)
        else:
            balance_argmax = self.dist.sample()
        trt_logs = torch.log(trt_counts)
        ref_logs = torch.log(ref_counts)
        trt_parts = trt_logs * balance_argmax
        ref_parts = ref_logs * balance_argmax
        trt_X = trt_parts[NUM].mean() - trt_parts[DENOM].mean()
        ref_X = ref_parts[NUM].mean() - ref_parts[DENOM].mean()
        # conditional logistic regression
        log_prob = self.beta_c * trt_X + self.beta_b * batch_id + ofs 
        log_prob -= torch.logsumexp(self.beta_c * trtX + self.beta_b * batch_id + ofs,
                                    self.beta * refX + self.beta_b * batch_id + ofs)
        return log_prob
        
    def training_step(self, batch, batch_idx):
        trt, ref, batch_ids = [x.to(self.device) for x in batch]
        log_prob = self.forward(trt, ref, batch_ids)
        loss = - torch.sum(log_prob)        
        current_lr = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]
        tensorboard_logs = {'train_loss' : loss, 'lr' : current_lr}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            trt, ref, batch_ids = [x.to(self.device) for x in batch]
            log_prob = self.forward(trt, ref, batch_ids)
            loss = - torch.sum(log_prob)
            
            pred = self.forward(trt, ref, batch_ids, hard=True)
            separation = torch.mean(pred > 0)
            
            ref_pred = self.forward(ref[::-1], ref, batch_ids, hard=True)
            acc = torch.mean(pred > ref_pred)
            
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
            losses = np.mean(list(map(lambda x: x['log'][m], outputs)))
            self.logger.experiment.add_scalar(m, meas, self.global_step)
            self.log(m, meas)
            tensorboard_logs[m] = meas
        return {'val_loss': tensorboard_logs['val_loss'],
                'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.vae.parameters(), lr=self.hparams['learning_rate'],
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
            required=False, type=str, default=1)
        parser.add_argument(
            '--learning-rate', help='Learning rate',
            required=False, type=float, default=1e-3)
        return parser
