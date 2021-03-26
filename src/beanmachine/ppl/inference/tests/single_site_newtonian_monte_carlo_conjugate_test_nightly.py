# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.testlib.abstract_conjugate import AbstractConjugateTests
from beanmachine.ppl.world import TransformType


class SingleSiteNewtonianMonteCarloConjugateTest(
    unittest.TestCase, AbstractConjugateTests
):
    @unittest.skip("Known to fail. Investigating in T77865889.")
    def test_beta_binomial_conjugate_run(self):
        nw = bm.SingleSiteNewtonianMonteCarlo()
        self.beta_binomial_conjugate_run(nw, num_samples=1000)
        # failing for transform proposer because hessian is extremely close to 0
        # NMC has a covariance that is too large to produce good samples
        # TODO: Add test case with TransformType.NONE

    def test_gamma_gamma_conjugate_run(self):
        nw_transform = bm.SingleSiteNewtonianMonteCarlo()
        self.gamma_gamma_conjugate_run(nw_transform, num_samples=200)

    def test_gamma_normal_conjugate_run(self):
        nw = bm.SingleSiteNewtonianMonteCarlo()
        self.gamma_normal_conjugate_run(nw, num_samples=600)

    def test_normal_normal_conjugate_run(self):
        nw = bm.SingleSiteNewtonianMonteCarlo()
        self.normal_normal_conjugate_run(nw, num_samples=500)

    # Following had to be increased to 1600 to pass variance test
    def test_distant_normal_normal_conjugate_run(self):
        nw = bm.SingleSiteNewtonianMonteCarlo()
        self.distant_normal_normal_conjugate_run(nw, num_samples=1600)

    def test_dirichlet_categorical_conjugate_run(self):
        nw = bm.SingleSiteNewtonianMonteCarlo()
        self.dirichlet_categorical_conjugate_run(nw, num_samples=2000)
        nw_transform = bm.SingleSiteNewtonianMonteCarlo(TransformType.DEFAULT)
        self.dirichlet_categorical_conjugate_run(nw_transform, num_samples=2000)
