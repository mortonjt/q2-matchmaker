import unittest
import biom
import pandas as pd
import numpy as np
from q2_matchmaker._milp import ConditionalBalanceClassifier
from q2_matchmaker.dataset import BiomDataModule
from pytorch_lightning import Trainer


class TestMILP(unittest.TestCase):
    N = 20
    D = 10
    obs = [f'o{i}' for i in range(D)]
    sam = [f's{i}' for i in range(N)]
    table = biom.Table(np.random.randn(D, N), obs, sam)

    metadata = pd.DataFrame(
        {
            'batch': np.arange(20) % 3,
            'label': np.arange(20) % 2,
            'match': list(range(5)) + list(range(5)) + list(range(5)) + list(range(5)),
            'train': ['Train'] * 10 + ['Test'] * 10
        }
    )
    dm = BiomDataModule(table, metadata,
                        match_column='match',
                        label_column='label',
                        reference_label=0,
                        batch_column='batch',
                        train_column='train',
                        batch_size=5,
                        num_workers=1)
    model = ConditionalBalanceClassifier(D)
    trainer = Trainer(
        max_epochs=1,
        check_val_every_n_epoch=1)
    trainer.fit(model, dm)


if __name__ == '__main__':
    unittest.main()
