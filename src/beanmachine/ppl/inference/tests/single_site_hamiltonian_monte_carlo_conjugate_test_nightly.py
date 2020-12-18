# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.testlib.abstract_conjugate import AbstractConjugateTests


class SingleSiteHamiltonianMonteCarloConjugateTest(
    unittest.TestCase, AbstractConjugateTests
):
    # TODO: Delta below should be reduced
    @unittest.skip("Known to fail. Investigating in T77865889.")
    def test_beta_binomial_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(0.5, 0.05)
        self.beta_binomial_conjugate_run(hmc, num_samples=1000, delta=0.55)

    # TODO: Delta below should be reduced
    @unittest.skip("Known to fail. Investigating in T77865889.")
    def test_gamma_gamma_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(0.1, 0.01)
        self.gamma_gamma_conjugate_run(hmc, num_samples=1000, delta=0.4)

    @unittest.skip("Known to fail. Investigating in T77865889.")
    def test_gamma_normal_conjugate_run(self):
        # see N209164
        hmc = bm.SingleSiteHamiltonianMonteCarlo(1.0)
        # division by 0 caused by first step size being too large
        # proposing a constrained of 0.0 in the halfspace
        self.gamma_gamma_conjugate_run(hmc, num_samples=2000, delta=0.44)

    def test_normal_normal_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(1.0, 0.05)
        self.normal_normal_conjugate_run(hmc, num_samples=500, delta=0.2)

    def test_distant_normal_normal_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(1.0, 0.1)
        self.distant_normal_normal_conjugate_run(hmc, num_samples=1000, delta=0.2)

    # TODO: Delta below should be reduced
    @unittest.skip("Known to fail. Investigating in T77865889.")
    def test_dirichlet_categorical_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(0.1, 0.01)
        self.dirichlet_categorical_conjugate_run(hmc, num_samples=10000, delta=1.0)
