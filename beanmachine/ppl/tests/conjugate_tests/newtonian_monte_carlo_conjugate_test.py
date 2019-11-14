# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from beanmachine.ppl.inference.single_site_newtonian_monte_carlo import (
    SingleSiteNewtonianMonteCarlo,
)
from beanmachine.ppl.tests.conjugate_tests.abstract_conjugate import (
    AbstractConjugateTests,
)


class SingleSiteNewtonianMonteCarloConjugateTest(
    unittest.TestCase, AbstractConjugateTests
):
    def test_beta_binomial_conjugate_run(self):
        nw = SingleSiteNewtonianMonteCarlo()
        self.beta_binomial_conjugate_run(nw, num_samples=150, delta=0.15)
        nw = SingleSiteNewtonianMonteCarlo(True)
        self.beta_binomial_conjugate_run(nw, num_samples=250, delta=0.15)

    def test_gamma_gamma_conjugate_run(self):
        nw_transform = SingleSiteNewtonianMonteCarlo()
        self.gamma_gamma_conjugate_run(nw_transform, num_samples=200, delta=0.15)

    def test_gamma_normal_conjugate_run(self):
        nw = SingleSiteNewtonianMonteCarlo()
        self.gamma_normal_conjugate_run(nw, num_samples=600, delta=0.15)

    def test_normal_normal_conjugate_run(self):
        nw = SingleSiteNewtonianMonteCarlo()
        self.normal_normal_conjugate_run(nw, num_samples=500, delta=0.15)

    def test_distant_normal_normal_conjugate_run(self):
        nw = SingleSiteNewtonianMonteCarlo()
        self.distant_normal_normal_conjugate_run(nw, num_samples=800, delta=0.15)

    def test_dirichlet_categorical_conjugate_run(self):
        nw = SingleSiteNewtonianMonteCarlo()
        self.dirichlet_categorical_conjugate_run(nw, num_samples=100, delta=0.15)
        nw_transform = SingleSiteNewtonianMonteCarlo(True)
        self.dirichlet_categorical_conjugate_run(
            nw_transform, num_samples=100, delta=0.15
        )
