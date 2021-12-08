# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.testlib.abstract_conjugate import AbstractConjugateTests


class SingleSiteHamiltonianMonteCarloConjugateTest(
    unittest.TestCase, AbstractConjugateTests
):
    def test_beta_binomial_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(0.5, 0.05)
        self.beta_binomial_conjugate_run(hmc, num_samples=500, num_adaptive_samples=500)

    def test_gamma_gamma_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(0.1, 0.01)
        self.gamma_gamma_conjugate_run(hmc, num_samples=500, num_adaptive_samples=500)

    def test_gamma_normal_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(0.5, 0.05)
        self.gamma_gamma_conjugate_run(hmc, num_samples=500, num_adaptive_samples=500)

    def test_normal_normal_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(1.0, 0.05)
        self.normal_normal_conjugate_run(hmc, num_samples=500, num_adaptive_samples=500)

    def test_dirichlet_categorical_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(0.1, 0.01)
        self.dirichlet_categorical_conjugate_run(
            hmc, num_samples=500, num_adaptive_samples=500
        )
