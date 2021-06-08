# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.testlib.abstract_conjugate import AbstractConjugateTests


class SingleSiteAdaptiveHamiltonianMonteCarloConjugateTest(
    unittest.TestCase, AbstractConjugateTests
):
    # for some transformed spaces, the acceptance ratio is never .65
    # see Adapative HMC Conjugate Tests for more details
    @unittest.skip("Known to fail. Investigating in T77865889.")
    def test_beta_binomial_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(1.0)
        self.beta_binomial_conjugate_run(hmc, num_samples=2000)

    @unittest.skip("Known to fail. Investigating in T77865889.")
    def test_gamma_gamma_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(1.0)
        self.gamma_normal_conjugate_run(hmc, num_samples=2000)

    @unittest.skip("Known to fail. Investigating in T77865889.")
    def test_gamma_normal_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(1.0)
        # division by 0 caused by first step size being too large
        # proposing a constrained of 0.0 in the halfspace
        self.gamma_gamma_conjugate_run(hmc, num_samples=2000)

    # TODO: The following test fails for higher n, namely 1-2K.
    #       This should be investigated further
    def test_normal_normal_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(0.5)
        self.normal_normal_conjugate_run(hmc, num_samples=200, num_adaptive_samples=50)

    @unittest.skip("Known to fail. Investigating in T77865889.")
    def test_dirichlet_categorical_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(0.1)
        self.dirichlet_categorical_conjugate_run(hmc, num_samples=1000)
