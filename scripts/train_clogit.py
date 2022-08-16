import os
import biom
import pandas as pd
from q2_matchmaker._milp import ConditionalBalanceClassifier
from q2_matchmaker.dataset import BiomDataModule, add_data_specific_args
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import yaml
import argparse


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
    # TODO : enable init probs later
    model = ConditionalBalanceClassifier(D, C, init_probs=None,
                                         temp=0.1, learning_rate=args.learning_rate)
    ckpt_path = os.path.join(args.output_directory, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        period=1,
        monitor='val_loss',
        mode='min',
        verbose=True)
    os.mkdir(args.output_directory)
    tb_logger = pl_loggers.TensorBoardLogger(f'{args.output_directory}/logs/')
    # save hyper-parameters to yaml file
    with open(f'{args.output_directory}/hparams.yaml', 'w') as outfile:
        yaml.dump(model._hparams, outfile, default_flow_style=False)


    trainer = Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        stochastic_weight_avg=False,
        check_val_every_n_epoch=10,
        gradient_clip_val=args.grad_clip,
        logger=tb_logger,
        callbacks=[checkpoint_callback])

    trainer.fit(model, dm)
    trainer.save_checkpoint(
        args.output_directory + '/last_ckpt.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser = ConditionalBalanceClassifier.add_model_specific_args(
        parser, add_help=False)
    parser = add_data_specific_args(parser, add_help=False)
    args = parser.parse_args()
    main(args)
