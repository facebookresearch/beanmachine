# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import scipy.stats
import torch
import torch.distributions as dist
import beanmachine.ppl as bm
from beanmachine.ppl.experimental.vi.VariationalInfer import (
    VariationalApproximation, MeanFieldVariationalInference
)
from beanmachine.ppl.model.statistical_model import StatisticalModel
from beanmachine.ppl.experimental.tests.vi.neals_funnel import NealsFunnel


class VariationalInferTest(unittest.TestCase):
    class NormalNormal:
        @bm.random_variable
        def mu(self):
            return dist.Normal(0, 10)

        @bm.random_variable
        def x(self, i):
            return dist.Normal(self.mu(), 1)

    def tearDown(self) -> None:
        StatisticalModel.reset()

    def test_neals_funnel(self):
        nf = NealsFunnel()

        vi = VariationalApproximation(
            target_log_prob=nf.log_prob,
            base_dist=dist.Independent(
                dist.Normal(torch.tensor([0., 0.]), torch.tensor([1., 1.])), 
                1))
        vi.train(epochs=300)

        # compare 1D marginals of empirical distributions using 2-sample K-S test
        nf_samples = nf.sample((20,)).squeeze().numpy()
        vi_samples = vi.sample((20,)).detach().numpy()

        self.assertGreaterEqual(
            scipy.stats.ks_2samp(nf_samples[:, 0], vi_samples[:, 0]).pvalue,
            0.05
        )
        self.assertGreaterEqual(
            scipy.stats.ks_2samp(nf_samples[:, 1], vi_samples[:, 1]).pvalue,
            0.05
        )

    def test_normal_normal(self):
        model = self.NormalNormal()
        vi = MeanFieldVariationalInference()
        vi_dicts = vi.infer(
            queries=[model.mu()],
            observations={
                model.x(1): torch.tensor(9.0),
                model.x(2): torch.tensor(10.0),
            },
            num_iter=100,
            lr=1e-1,
        )

        mu_approx = vi_dicts[model.mu()]
        sample_mean = mu_approx.sample((100, 1)).mean()
        self.assertGreater(sample_mean, 8.0)

        sample_var = mu_approx.sample((100, 1)).var()
        self.assertGreater(sample_var, 0.3)

    def test_normal_normal_studentt_base_dist(self):
        model = self.NormalNormal()
        vi = MeanFieldVariationalInference()
        vi_dicts = vi.infer(
            queries=[model.mu()],
            observations={
                model.x(1): torch.tensor(9.0),
                model.x(2): torch.tensor(10.0),
            },
            num_iter=100,
            lr=1e-1,
            base_dist=dist.StudentT(df=1),
        )

        mu_approx = vi_dicts[model.mu()]
        sample_mean = mu_approx.sample((100, 1)).mean()
        self.assertGreater(sample_mean, 8.0)

        sample_var = mu_approx.sample((100, 1)).var()
        self.assertGreater(sample_var, 0.3)