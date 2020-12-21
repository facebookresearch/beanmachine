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
        self.beta_binomial_conjugate_run(hmc, num_samples=2000, delta=0.25)

    @unittest.skip("Known to fail. Investigating in T77865889.")
    def test_gamma_gamma_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(1.0)
        self.gamma_normal_conjugate_run(hmc, num_samples=2000, delta=0.99)

    @unittest.skip("Known to fail. Investigating in T77865889.")
    def test_gamma_normal_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(1.0)
        # division by 0 caused by first step size being too large
        # proposing a constrained of 0.0 in the halfspace
        self.gamma_gamma_conjugate_run(hmc, num_samples=2000, delta=0.44)

    def test_normal_normal_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(0.5)
        self.normal_normal_conjugate_run(
            hmc, num_samples=200, delta=0.15, num_adaptive_samples=200
        )

    def test_distant_normal_normal_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(1.0)
        self.distant_normal_normal_conjugate_run(hmc, num_samples=1000, delta=0.15)

    @unittest.skip("Known to fail. Investigating in T77865889.")
    def test_dirichlet_categorical_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(0.1)
        # TODO: The delta in the following needs to be reduced
        self.dirichlet_categorical_conjugate_run(hmc, num_samples=1000, delta=0.5)
