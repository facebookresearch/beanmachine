# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.testlib.abstract_conjugate import AbstractConjugateTests


class SingleSiteAncestralMetropolisHastingsConjugateTest(
    unittest.TestCase, AbstractConjugateTests
):
    def setUp(self):
        self.mh = bm.SingleSiteAncestralMetropolisHastings()

    def test_beta_binomial_conjugate_run(self):
        self.beta_binomial_conjugate_run(self.mh)

    def test_gamma_gamma_conjugate_run(self):
        self.gamma_gamma_conjugate_run(self.mh)

    def test_gamma_normal_conjugate_run(self):
        # Converges with 10k and more iterations but will use a bigger delta for
        # now to have a faster test.
        self.gamma_normal_conjugate_run(self.mh, num_samples=20000, delta=0.2)

    def test_normal_normal_conjugate_run(self):
        # Converges with 10k and more iterations but will use a bigger delta for
        # now to have a faster test.
        self.normal_normal_conjugate_run(self.mh, num_samples=5000, delta=0.1)

    # Ridiculous detla due to failing convergence
    # The delta should be unnecessary after new testing is in place
    @unittest.skip("Expect to fail. N_eff is 5 @ 1K sample size.")
    def test_distant_normal_normal_conjugate_run(self):
        # We don't expect ancestral to be able to converge fast for this model.
        self.distant_normal_normal_conjugate_run(self.mh, num_samples=1000, delta=95)

    def test_dirichlet_categorical_conjugate_run(self):
        self.dirichlet_categorical_conjugate_run(self.mh, num_samples=10000, delta=0.1)
