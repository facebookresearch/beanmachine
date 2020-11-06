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
    def tearDown(self) -> None:
        StatisticalModel.reset()

    def test_neals_funnel(self):
        nf = NealsFunnel()

        vi = VariationalApproximation(target_log_prob=nf.log_prob, d=2)
        vi.train(epochs=300)

        # compare 1D marginals of empirical distributions using 2-sample K-S test
        nf_samples = nf.sample(sample_shape=(30, 2)).squeeze().numpy()
        vi_samples = vi.sample((30, 2)).numpy()

        self.assertTrue(
            scipy.stats.ks_2samp(nf_samples[:, 0], vi_samples[:, 0]).pvalue >= 0.05
        )
        self.assertTrue(
            scipy.stats.ks_2samp(nf_samples[:, 1], vi_samples[:, 1]).pvalue >= 0.05
        )

    def test_normal_normal(self):
        @bm.random_variable
        def mu():
            return dist.Normal(0, 10)

        @bm.random_variable
        def x(i):
            return dist.Normal(mu(), 1)

        vi = MeanFieldVariationalInference()
        vi_dicts = vi.infer(
            queries=[mu()],
            observations={
                x(1): torch.tensor(9.0),
                x(2): torch.tensor(10.0),
            },
            num_iter=100,
            lr=1e-1,
        )

        mu_approx = vi_dicts[mu()]
        sample_mean = mu_approx.sample((100, 1)).mean()
        print(sample_mean)
        self.assertTrue(sample_mean > 2.0)
