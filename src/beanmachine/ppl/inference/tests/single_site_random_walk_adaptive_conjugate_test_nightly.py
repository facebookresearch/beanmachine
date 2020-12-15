# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.testlib.abstract_conjugate import AbstractConjugateTests


class SingleSiteAdaptiveRandomWalkConjugateTest(
    unittest.TestCase, AbstractConjugateTests
):
    def setUp(self):
        self.mh = bm.SingleSiteRandomWalk(step_size=5.0)

    def test_beta_binomial_conjugate_run(self):
        self.mh = bm.SingleSiteRandomWalk(step_size=1.0)
        self.beta_binomial_conjugate_run(
            # Delta changed to 0.6 after RV diff. This is a big jump from 0.2
            self.mh,
            num_samples=3000,
            num_adaptive_samples=1600,
            delta=0.6,
        )
        # self.assertTrue(False)

    def test_gamma_gamma_conjugate_run(self):
        self.mh = bm.SingleSiteRandomWalk(step_size=3.0)
        self.gamma_gamma_conjugate_run(
            self.mh, num_samples=2000, num_adaptive_samples=500, delta=0.2
        )

    def test_gamma_normal_conjugate_run(self):
        self.mh = bm.SingleSiteRandomWalk(step_size=5.0)
        self.gamma_normal_conjugate_run(
            self.mh, num_samples=2000, num_adaptive_samples=1000, delta=0.2
        )

    def test_normal_normal_conjugate_run(self):
        self.normal_normal_conjugate_run(
            self.mh, num_samples=1000, num_adaptive_samples=500, delta=0.2
        )

    def test_distant_normal_normal_conjugate_run(self):
        # TODO: The delta in the following needs to be reduced
        self.normal_normal_conjugate_run(
            self.mh, num_samples=4000, num_adaptive_samples=500, delta=0.13
        )

    def test_dirichlet_categorical_conjugate_run(self):
        self.dirichlet_categorical_conjugate_run(
            self.mh, num_samples=2000, num_adaptive_samples=500, delta=0.2
        )
