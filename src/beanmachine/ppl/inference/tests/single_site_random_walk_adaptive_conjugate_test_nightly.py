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
            self.mh, num_samples=3000, num_adaptive_samples=1600
        )

    def test_gamma_gamma_conjugate_run(self):
        self.mh = bm.SingleSiteRandomWalk(step_size=3.0)
        self.gamma_gamma_conjugate_run(
            self.mh, num_samples=5000, num_adaptive_samples=7000
        )

    def test_gamma_normal_conjugate_run(self):
        self.mh = bm.SingleSiteRandomWalk(step_size=5.0)
        self.gamma_normal_conjugate_run(
            self.mh, num_samples=6000, num_adaptive_samples=5000
        )

    @unittest.skip("FIXME: Proposer getting stuck.")
    def test_normal_normal_conjugate_run(self):
        self.normal_normal_conjugate_run(
            self.mh, num_samples=2000, num_adaptive_samples=2000
        )

    # TODO: Expected n_eff levels should be documented in tests
    def test_dirichlet_categorical_conjugate_run(self):
        self.dirichlet_categorical_conjugate_run(
            self.mh, num_samples=2000, num_adaptive_samples=2000
        )
