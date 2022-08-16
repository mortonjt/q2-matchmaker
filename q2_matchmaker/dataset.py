import biom
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse


class BiomDataset(Dataset):
    """Loads a `.biom` file.

    Parameters
    ----------
    filename : Path
        Filepath to biom table
    metadata_file : Path
        Filepath to sample metadata
    batch_category : str
        Column name for batch indices

    Notes
    -----
    The name Classification_Group is protected. Do not pass in a column
    with this name.
    """
    def __init__(
            self,
            table: biom.Table,
            metadata: pd.DataFrame,
            batch_column: str,
            label_column: str,
            reference_label: str,
            match_column: str,
            dir_boot: bool = True,
            pseudocount: int = 1):
        super(BiomDataset).__init__()
        if np.any(table.sum(axis='sample') <= 0):
            ValueError('Biom table has zero counts.')
        self.table = table
        self.metadata = metadata
        self.batch_column = batch_column
        self.match_column = match_column
        self.label_column = label_column
        self.dir_boot = dir_boot
        self.pc = pseudocount

        if 'Classification_Group' in [batch_column, match_column,
                                      label_column]:
            raise ValueError('Classification_Group is a reserved keyword. '
                             'Double check your metadata.')

        # match the metadata with the table
        ids = set(self.table.ids()) & set(self.metadata.index)
        filter_f = lambda v, i, m: i in ids
        self.table = self.table.filter(filter_f, axis='sample')
        self.metadata = self.metadata.loc[self.table.ids()]

        # sort by match ids and labels
        cats = self.metadata[self.label_column] == reference_label
        self.metadata['Classification_Group'] = cats
        self.metadata = self.metadata.sort_values(
            by=[self.label_column, self.match_column])

        if self.metadata.index.name is None:
            raise ValueError('`Index` must have a name either'
                             '`sampleid`, `sample-id` or #SampleID')
        self.index_name = self.metadata.index.name
        #self.metadata = self.metadata.reset_index()

        batch_cats = self.metadata[self.batch_column].unique()
        self.batch_cats = pd.Series(
            np.arange(len(batch_cats)), index=batch_cats)
        self.batch_indices = np.array(
            list(map(lambda x: self.batch_cats.loc[x],
                     self.metadata[self.batch_column].values)))
        # match ids
        self.matchings = pd.Series(
            self.metadata[self.match_column].unique(),
            index=np.arange(len(self.metadata[self.match_column].unique())))
        self.m_dict = dict(list(self.metadata.groupby(self.match_column)))
        self.Ndata = len(self.matchings)

    def __len__(self) -> int:
        return len(self.matchings)

    def __getitem__(self, i : int):
        """ Returns all of the samples for a given matching

        Returns
        -------
        trt_counts : np.array
            OTU counts for treatment group
        ref_counts : np.array
            OTU counts for reference group
        batch_indices : np.array
            Membership ids for batch samples. If not specified, return None.
        """
        pair_md = self.m_dict[self.matchings.loc[i]]
        trt_idx, ref_idx = pair_md.index[0], pair_md.index[1]
        ref_counts = self.table.data(id=ref_idx, axis='sample') + self.pc
        trt_counts = self.table.data(id=trt_idx, axis='sample') + self.pc
        batch_indices = self.batch_indices[i]
        if self.dir_boot:
            trt_counts = np.random.dirichlet(trt_counts)
            ref_counts = np.random.dirichlet(ref_counts)

        return trt_counts, ref_counts, batch_indices

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        start = 0
        end = self.__len__()

        if worker_info is None:  # single-process data loading
            for i in range(end):
                yield self.__getitem__(i)
        else:
            worker_id = worker_info.id
            w = float(worker_info.num_workers)
            t = (end - start)
            w = float(worker_info.num_workers)
            per_worker = int(math.ceil(t / w))
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)
            for i in range(iter_start, iter_end):
                yield self.__getitem__(i)

def collate_match_f(batch):

    trt_counts_list = np.vstack([b[0] for b in batch])
    ref_counts_list = np.vstack([b[1] for b in batch])

    batch_ids = np.vstack([b[2] for b in batch])

    trt_counts = torch.from_numpy(trt_counts_list).float()
    ref_counts = torch.from_numpy(ref_counts_list).float()
    batch_ids = torch.from_numpy(batch_ids).long()
    return trt_counts, ref_counts, batch_ids.squeeze()


class BiomDataModule(pl.LightningDataModule):
    """
    Notes
    -----
    Assumes that `train_column` has values True and False
    """
    def __init__(self, biom_table : biom.Table,
                 metadata : pd.DataFrame,
                 match_column : str,
                 label_column : str,
                 reference_label : str,
                 batch_column : str,
                 train_column : str,
                 batch_size : int = 10,
                 num_workers : int = 1):
        super().__init__()
        self.metadata = metadata
        self.match_column = match_column
        self.batch_column = batch_column
        self.label_column = label_column
        self.reference_label = reference_label

        train_idx = set(metadata.loc[metadata[train_column] == 'Train'].index)
        test_idx = set(metadata.loc[metadata[train_column] == 'Test'].index)
        train_filter = lambda v, i, m: i in train_idx
        val_filter = lambda v, i, m: i in test_idx

        self.train_biom = biom_table.filter(train_filter, axis='sample', inplace=False)
        self.val_biom = biom_table.filter(val_filter, axis='sample', inplace=False)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_f = collate_match_f

    def train_dataloader(self):
        train_dataset = BiomDataset(
            self.train_biom,
            metadata=self.metadata,
            batch_column=self.batch_column,
            label_column=self.label_column,
            match_column=self.match_column,
            reference_label=self.reference_label
        )

        batch_size = min(len(train_dataset) // 2 - 1, self.batch_size)
        assert batch_size > 1
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size,
            collate_fn=self.collate_f, shuffle=True,
            num_workers=self.num_workers, drop_last=True,
            pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = BiomDataset(
            self.val_biom,
            metadata=self.metadata,
            batch_column=self.batch_column,
            label_column=self.label_column,
            match_column=self.match_column,
            reference_label=self.reference_label)
        batch_size = min(len(val_dataset) // 2 - 1, self.batch_size)
        assert batch_size > 1
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size,
            collate_fn=self.collate_f, shuffle=False,
            num_workers=self.num_workers, drop_last=True,
            pin_memory=True)
        return val_dataloader


def add_data_specific_args(parent_parser, add_help=True):
    parser = argparse.ArgumentParser(parents=[parent_parser],
                                     add_help=add_help)
    # Arguments specific for dataloaders
    parser.add_argument(
        '--biom-table', help='Biom table file', required=True)
    parser.add_argument(
        '--sample-metadata', help='Sample metadata file', required=True)
    parser.add_argument(
        '--batch-column',
        help='Sample metadata column for batch effects.',
        required=True, type=str)
    parser.add_argument(
        '--train-column',
        help=('Specifies training and testing samples '
              '(must be labeled `train` and `test`).'),
        required=True, type=str)
    parser.add_argument(
        '--label-column',
        help='Sample metadata column for class predictions.',
        required=True, type=str)
    parser.add_argument(
        '--reference-label',
        help='Reference category for classes.',
        required=True, type=str)
    parser.add_argument(
        '--match-column',
        help='Sample metadata column for match ids.',
        required=True, type=str)
    parser.add_argument(
        '--dir-boot',
        help='Specifies if dirichilet bootstrapping is used..',
        action='store_true',
        required=True, type=bool)
    parser.set_defaults(dir_boot=True)
    parser.add_argument(
        '--pseudocount',
        help='Pseudocount',
        required=False, type=int, default=1)
    parser.add_argument(
        '--batch-size', help='Training batch size',
        required=False, type=int, default=32)
    # Arguments specific for trainer
    parser.add_argument(
        '--epochs',
        help='Number of epochs (aka iterations) to train model.',
        required=False, type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--output-directory', type=str, default=None)
    return parser
