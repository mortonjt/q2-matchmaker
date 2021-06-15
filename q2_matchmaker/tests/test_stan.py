import unittest
import numpy as np
from q2_matchmaker._stan import (
<<<<<<< HEAD
    _case_control_sim, _case_control_full,
    _case_control_data, _case_control_single,
)
from biom import Table
from skbio.stats.composition import alr_inv, clr
import arviz as az


class TestCaseControl(unittest.TestCase):
=======
    _case_control_sim,
    NegativeBinomialCaseControl
)
from biom import Table
from birdman.diagnostics import r2_score
import arviz as az
from dask.distributed import Client, LocalCluster
from scipy.stats import pearsonr


class TestNegativeBinomialCaseControl(unittest.TestCase):
>>>>>>> c90db5ad2200aac412c3b8ec9c5c8f46ef056b7e

    def setUp(self):
        np.random.seed(0)
        self.table, self.metadata, self.diff = _case_control_sim(
<<<<<<< HEAD
            n=50, d=4, depth=100)

    def test_case_control_full(self):
        # fit once
        sm, posterior = _case_control_full(
            self.table.values,
            case_ctrl_ids=self.metadata['reps'].values,
            case_member=self.metadata['diff'].values,
            depth=self.table.sum(axis=1),
            mc_samples=100)
        dat = _case_control_data(self.table.values,
                                 case_ctrl_ids=self.metadata['reps'].values,
                                 case_member=self.metadata['diff'].values,
                                 depth=self.table.sum(axis=1))
        gen = sm.generate_quantities(
            data=dat, mcmc_sample=posterior)
        gen_table = gen.generated_quantities[0].reshape((50, 4)) + 1
        # refit to see if the parameters can be recovered
        # from the generated data
        _, re_posterior = _case_control_full(
            gen_table,
            case_ctrl_ids=self.metadata['reps'].values,
            case_member=self.metadata['diff'].values,
            depth=self.table.sum(axis=1),
            mc_samples=1000)

        # TODO: test with random initialization
        res_diff = re_posterior.stan_variable('diff')
        exp_diff = posterior.stan_variable('diff').mean(0)
        rm = res_diff.mean(0)
        rs = res_diff.std(0)
        for i in range(len(self.diff)):
            self.assertTrue(
                (rm[i] - 3 * rs[i]) <= exp_diff[i] and
                (exp_diff[i] <= (rm[i] + 3 * rs[i]))
            )


class TestCaseControlSingle(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.table, self.metadata, self.diff = _case_control_sim(
            n=50, d=4, depth=100)
        self.diff = clr(alr_inv(self.diff))

    def test_cc_full(self):
        for i in range(self.table.shape[1]):
            res = _case_control_single(
                self.table.values[:, i],
                case_ctrl_ids=self.metadata['reps'].values,
                case_member=self.metadata['diff'].values,
                depth=self.table.sum(axis=1),
                mc_samples=500)
            rm = res['posterior']['diff'].mean()
            rs = res['posterior']['diff'].std()
            self.assertTrue(
                (rm - 2 * rs) <= self.diff[i] and
                (self.diff[i] <= (rm + 2 * rs))
            )
=======
            n=100, d=20, depth=1000)

    def test_cc_diagnostics(self):
        biom_table = Table(self.table.values.T,
                           list(self.table.columns),
                           list(self.table.index))

        nb = NegativeBinomialCaseControl(
            table=biom_table,
            matching_column="reps",
            status_column="diff",
            metadata=self.metadata,
            reference_status='1',
            chains=1,
            seed=42)
        nb.compile_model()
        dask_args = {'n_workers': 1, 'threads_per_worker': 1}
        cluster = LocalCluster(**dask_args)
        cluster.scale(dask_args['n_workers'])
        Client(cluster)
        nb.fit_model()
        inf = nb.to_inference_object()
        self.assertEqual(inf['posterior']['mu'].shape, (10, 1, 1000))

        res = r2_score(inf)
        self.assertGreater(res['r2'], 0.3)
        az.loo(inf)
        az.bfmi(inf)
        az.rhat(inf, var_names=nb.param_names)
        az.ess(inf, var_names=nb.param_names)

    def test_cc_fit(self):
        biom_table = Table(self.table.values.T,
                           list(self.table.columns),
                           list(self.table.index))
        nb = NegativeBinomialCaseControl(
            table=biom_table,
            matching_column="reps",
            status_column="diff",
            metadata=self.metadata,
            reference_status='1',
            mu_scale=1,
            sigma_scale=.1,
            disp_scale=0.01,
            # priors specific to the simulation
            control_loc=-6,
            control_scale=3,
            chains=1,
            seed=42)
        nb.compile_model()
        dask_args = {'n_workers': 1, 'threads_per_worker': 1}
        cluster = LocalCluster(**dask_args)
        cluster.scale(dask_args['n_workers'])
        Client(cluster)
        nb.fit_model()
        samples = nb.to_inference_object()
        exp = np.array([0] + list(self.diff))
        res = samples['posterior']['diff'].mean(dim=['chain', 'draw'])
        r, p = pearsonr(res, exp)
        self.assertGreater(r, 0.3)
        self.assertLess(p, 0.05)
>>>>>>> c90db5ad2200aac412c3b8ec9c5c8f46ef056b7e


if __name__ == '__main__':
    unittest.main()
