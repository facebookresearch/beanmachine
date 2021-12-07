# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from beanmachine.ppl.inference.single_site_nmc import (
    SingleSiteNewtonianMonteCarlo,
)
from beanmachine.ppl.testlib.abstract_conjugate import AbstractConjugateTests


class SingleSiteNewtonianMonteCarloConjugateTest(
    unittest.TestCase, AbstractConjugateTests
):
    # TODO: Decrease the num_samples; num_samples>=2000 to get n_eff>=30 is
    # unreasonable. It currently fails for num_samples=1000 because because
    # hessian (for transform proposer) is extremely close to 0

    def test_beta_binomial_conjugate_run(self):
        nw = SingleSiteNewtonianMonteCarlo()
        self.beta_binomial_conjugate_run(nw, num_samples=2000)

    def test_gamma_gamma_conjugate_run(self):
        nw_transform = SingleSiteNewtonianMonteCarlo()
        self.gamma_gamma_conjugate_run(nw_transform, num_samples=200)

    def test_gamma_normal_conjugate_run(self):
        nw = SingleSiteNewtonianMonteCarlo()
        self.gamma_normal_conjugate_run(nw, num_samples=600)

    def test_normal_normal_conjugate_run(self):
        nw = SingleSiteNewtonianMonteCarlo()
        self.normal_normal_conjugate_run(nw, num_samples=500)

    def test_dirichlet_categorical_conjugate_run(self):
        nw = SingleSiteNewtonianMonteCarlo()
        self.dirichlet_categorical_conjugate_run(nw, num_samples=2000)
