# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import scipy.stats
from beanmachine.ppl.experimental.vi.VariationalInfer import VariationalApproximation
from beanmachine.ppl.model.statistical_model import StatisticalModel
from beanmachine.ppl.experimental.tests.vi.neals_funnel import NealsFunnel


class VariationalInferTest(unittest.TestCase):
    def tearDown(self) -> None:
        StatisticalModel.reset()

    def test_neals_funnel(self):
        nf = NealsFunnel()

        vi = VariationalApproximation(target=nf)
        vi.train(epochs=1000)

        # compare 1D marginals of empirical distributions using 2-sample K-S test
        nf_samples = nf.sample(sample_shape=(100, 2)).squeeze().numpy()
        vi_samples = vi.sample((100, 2)).numpy()

        self.assertTrue(
            scipy.stats.ks_2samp(nf_samples[:, 0], vi_samples[:, 0]).pvalue >= 0.05
        )
        self.assertTrue(
            scipy.stats.ks_2samp(nf_samples[:, 1], vi_samples[:, 1]).pvalue >= 0.05
        )
