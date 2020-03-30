# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from beanmachine.ppl.inference.single_site_random_walk import SingleSiteRandomWalk
from beanmachine.ppl.tests.conjugate_tests.abstract_conjugate import (
    AbstractConjugateTests,
)


class SingleSiteAdaptiveRandomWalkConjugateTest(
    unittest.TestCase, AbstractConjugateTests
):
    def setUp(self):
        self.mh = SingleSiteRandomWalk(step_size=5.0)

    def test_beta_binomial_conjugate_run(self):
        self.mh = SingleSiteRandomWalk(step_size=1.0)
        self.beta_binomial_conjugate_run(
            self.mh, num_samples=3000, num_adapt_steps=1500, delta=0.2
        )
        # self.assertTrue(False)

    def test_gamma_gamma_conjugate_run(self):
        self.mh = SingleSiteRandomWalk(step_size=3.0)
        self.gamma_gamma_conjugate_run(
            self.mh, num_samples=2000, num_adapt_steps=500, delta=0.2
        )

    def test_gamma_normal_conjugate_run(self):
        self.mh = SingleSiteRandomWalk(step_size=5.0)
        self.gamma_normal_conjugate_run(
            self.mh, num_samples=2000, num_adapt_steps=1000, delta=0.2
        )

    def test_normal_normal_conjugate_run(self):
        self.normal_normal_conjugate_run(
            self.mh, num_samples=1000, num_adapt_steps=500, delta=0.2
        )

    def test_distant_normal_normal_conjugate_run(self):
        self.normal_normal_conjugate_run(
            self.mh, num_samples=4000, num_adapt_steps=500, delta=0.1
        )

    def test_dirichlet_categorical_conjugate_run(self):
        self.dirichlet_categorical_conjugate_run(
            self.mh, num_samples=2000, num_adapt_steps=500, delta=0.2
        )
