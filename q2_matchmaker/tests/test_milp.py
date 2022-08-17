import unittest
import biom
import pandas as pd
import numpy as np
from q2_matchmaker._milp import ConditionalBalanceClassifier, BalanceClassifier
from q2_matchmaker.dataset import BiomDataModule
from pytorch_lightning import Trainer


class TestMILP(unittest.TestCase):
    N = 40
    D = 10
    C = 3
    obs = [f'o{i}' for i in range(D)]
    sam = [f's{i}' for i in range(N)]
    table = biom.Table(np.exp(np.random.randn(D, N)).round(), obs, sam)

    metadata = pd.DataFrame(
        {
            'batch': np.arange(N) % 3,
            'label': np.arange(N) % 2,
            'match': list(range(N // 4)) * 4,
            'train': ['Train'] * (N // 2) + ['Test'] * (N // 2)
        }, index=sam
    )
    metadata.index.name = 'sampleid'
    dm = BiomDataModule(table, metadata,
                        match_column='match',
                        label_column='label',
                        reference_label=0,
                        batch_column='batch',
                        train_column='train',
                        batch_size=5,
                        num_workers=1)
    model = BalanceClassifier(D, C)
    trainer = Trainer(
        max_epochs=1,
        check_val_every_n_epoch=1)
    trainer.fit(model, dm)


if __name__ == '__main__':
    unittest.main()
