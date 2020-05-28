# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.tests.conjugate_tests.abstract_conjugate import (
    AbstractConjugateTests,
)


class SingleSiteNewtonianMonteCarloConjugateTest(
    unittest.TestCase, AbstractConjugateTests
):
    def test_beta_binomial_conjugate_run(self):
        nw = bm.SingleSiteNewtonianMonteCarlo()
        self.beta_binomial_conjugate_run(nw, num_samples=350, delta=0.15)
        # failing for transform proposer because hessian is extremely close to 0
        # NMC has a covariance that is too large to produce good samples

    def test_gamma_gamma_conjugate_run(self):
        nw_transform = bm.SingleSiteNewtonianMonteCarlo()
        self.gamma_gamma_conjugate_run(nw_transform, num_samples=200, delta=0.15)

    def test_gamma_normal_conjugate_run(self):
        nw = bm.SingleSiteNewtonianMonteCarlo()
        self.gamma_normal_conjugate_run(nw, num_samples=600, delta=0.15)

    def test_normal_normal_conjugate_run(self):
        nw = bm.SingleSiteNewtonianMonteCarlo()
        self.normal_normal_conjugate_run(nw, num_samples=500, delta=0.15)

    def test_distant_normal_normal_conjugate_run(self):
        nw = bm.SingleSiteNewtonianMonteCarlo()
        self.distant_normal_normal_conjugate_run(nw, num_samples=800, delta=0.15)

    def test_dirichlet_categorical_conjugate_run(self):
        nw = bm.SingleSiteNewtonianMonteCarlo()
        self.dirichlet_categorical_conjugate_run(nw, num_samples=100, delta=0.15)
        nw_transform = bm.SingleSiteNewtonianMonteCarlo(True)
        self.dirichlet_categorical_conjugate_run(
            nw_transform, num_samples=100, delta=0.15
        )
