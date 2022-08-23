import os
import biom
import torch
import arviz as az
import numpy as np
import pandas as pd
from q2_matchmaker._milp import BalanceClassifier, ConditionalBalanceClassifier
from q2_matchmaker.dataset import BiomDataModule, add_data_specific_args
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import yaml
import argparse


def preprocess_init(table, path, key='diff'):
    diffs = pd.read_csv(path, index_col=0)
    round_diffs = diffs.applymap(np.tanh).round()
    vc = round_diffs.apply(lambda x: x.value_counts(), axis=1)
    vc = vc.fillna(0) + 1
    vc = vc.apply(lambda x: np.log(x / x.sum()), axis=1)
    unobserved_ids = list(set(table.ids(axis='observation')) - set(vc.index))
    vc = vc.reindex(unobserved_ids)
    idx = pd.isnull(vc).sum(axis=1) == 3
    vc.loc[idx] = np.log([0.05, 0.9, 0.05])  # push prior towards neutral
    return vc


def main(args):
    table = biom.load_table(args.biom_table)
    metadata = pd.read_table(args.sample_metadata, index_col=0)
    C = len(metadata[args.batch_column].unique())
    D = table.shape[0]
    dm = BiomDataModule(table, metadata,
                        match_column=args.match_column,
                        label_column=args.label_column,
                        reference_label=args.reference_label,
                        batch_column=args.batch_column,
                        train_column=args.train_column,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers)
    if args.init_probs is not None:
        probs = torch.Tensor(preprocess_init(table, args.init_probs).values)
    else:
        probs = None

    # TODO : enable init probs later
    model = ConditionalBalanceClassifier(
        D, C, init_probs=probs,
        temp=0.1, learning_rate=args.learning_rate)

    # ckpt_path = os.path.join(args.output_directory, "checkpoints")
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=ckpt_path,
    #     period=1,
    #     monitor='val_loss',
    #     mode='min',
    #     verbose=True)

    os.mkdir(args.output_directory)
    tb_logger = pl_loggers.TensorBoardLogger(f'{args.output_directory}/logs/')
    # save hyper-parameters to yaml file
    with open(f'{args.output_directory}/hparams.yaml', 'w') as outfile:
        yaml.dump(model._hparams, outfile, default_flow_style=False)

    trainer = Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        # gradient_clip_val=10,
        logger=tb_logger,
        # callbacks=[checkpoint_callback]
    )

    trainer.fit(model, dm)
    trainer.save_checkpoint(
        args.output_directory + '/last_ckpt.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser = BalanceClassifier.add_model_specific_args(
        parser, add_help=False)
    parser = add_data_specific_args(parser, add_help=False)
    args = parser.parse_args()
    main(args)
