# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from beanmachine.ppl.inference.single_site_hamiltonian_monte_carlo import (
    SingleSiteHamiltonianMonteCarlo,
)
from beanmachine.ppl.tests.conjugate_tests.abstract_conjugate import (
    AbstractConjugateTests,
)


class SingleSiteHamiltonianMonteCarloConjugateTest(
    unittest.TestCase, AbstractConjugateTests
):
    def test_beta_binomial_conjugate_run(self):
        hmc = SingleSiteHamiltonianMonteCarlo(0.05, 10)
        self.beta_binomial_conjugate_run(hmc, num_samples=150, delta=0.15)

    def test_gamma_gamma_conjugate_run(self):
        hmc = SingleSiteHamiltonianMonteCarlo(0.01, 10)
        self.gamma_gamma_conjugate_run(hmc, num_samples=200, delta=0.15)

    def test_gamma_normal_conjugate_run(self):
        pass

    def test_normal_normal_conjugate_run(self):
        hmc = SingleSiteHamiltonianMonteCarlo(0.05, 10)
        self.normal_normal_conjugate_run(hmc, num_samples=500, delta=0.1)

    def test_distant_normal_normal_conjugate_run(self):
        hmc = SingleSiteHamiltonianMonteCarlo(0.1, 10)
        self.distant_normal_normal_conjugate_run(hmc, num_samples=500, delta=0.1)

    def test_dirichlet_categorical_conjugate_run(self):
        hmc = SingleSiteHamiltonianMonteCarlo(0.01, 10)
        self.dirichlet_categorical_conjugate_run(hmc, num_samples=200, delta=0.15)
