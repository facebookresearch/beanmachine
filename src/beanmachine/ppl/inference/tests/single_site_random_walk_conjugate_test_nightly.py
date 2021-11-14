# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from beanmachine.ppl.experimental.global_inference.single_site_random_walk import (
    SingleSiteRandomWalk,
)
from beanmachine.ppl.testlib.abstract_conjugate import AbstractConjugateTests


class SingleSiteRandomWalkConjugateTest(unittest.TestCase, AbstractConjugateTests):
    def setUp(self):
        self.mh = SingleSiteRandomWalk(step_size=1.0)

    def test_beta_binomial_conjugate_run(self):
        mh = SingleSiteRandomWalk(step_size=0.3)
        self.beta_binomial_conjugate_run(mh, num_samples=5000)

    def test_gamma_gamma_conjugate_run(self):
        self.gamma_gamma_conjugate_run(self.mh, num_samples=10000)

    def test_gamma_normal_conjugate_run(self):
        self.gamma_normal_conjugate_run(self.mh, num_samples=10000)

    def test_normal_normal_conjugate_run(self):
        mh = SingleSiteRandomWalk(step_size=1.5)
        self.normal_normal_conjugate_run(mh, num_samples=1000)

    def test_dirichlet_categorical_conjugate_run(self):
        self.dirichlet_categorical_conjugate_run(self.mh, num_samples=10000)
