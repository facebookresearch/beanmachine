# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from beanmachine.ppl.experimental.mala import (
    SingleSiteMetropolisAdapatedLangevinAlgorithm,
)
from beanmachine.ppl.testlib.abstract_conjugate import AbstractConjugateTests


class SingleSiteMetropolisAdapatedLangevinAlgorithmConjugateTest(
    unittest.TestCase, AbstractConjugateTests
):
    def test_beta_binomial_conjugate_run(self):
        mala = SingleSiteMetropolisAdapatedLangevinAlgorithm(0.05)
        self.beta_binomial_conjugate_run(
            mala, num_samples=500, num_adaptive_samples=500
        )

    def test_gamma_gamma_conjugate_run(self):
        mala = SingleSiteMetropolisAdapatedLangevinAlgorithm(0.05)
        self.gamma_gamma_conjugate_run(mala, num_samples=500, num_adaptive_samples=500)

    def test_gamma_normal_conjugate_run(self):
        mala = SingleSiteMetropolisAdapatedLangevinAlgorithm(0.05)
        self.gamma_normal_conjugate_run(mala, num_samples=500, num_adaptive_samples=500)

    def test_normal_normal_conjugate_run(self):
        mala = SingleSiteMetropolisAdapatedLangevinAlgorithm(0.05)
        self.normal_normal_conjugate_run(
            mala, num_samples=500, num_adaptive_samples=500
        )

    def test_dirichlet_categorical_conjugate_run(self):
        mala = SingleSiteMetropolisAdapatedLangevinAlgorithm(0.01)
        self.dirichlet_categorical_conjugate_run(
            mala, num_samples=500, num_adaptive_samples=500
        )
