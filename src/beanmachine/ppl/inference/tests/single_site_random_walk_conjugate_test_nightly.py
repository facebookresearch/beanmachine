# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.testlib.abstract_conjugate import AbstractConjugateTests


class SingleSiteRandomWalkConjugateTest(unittest.TestCase, AbstractConjugateTests):
    def setUp(self):
        self.mh = bm.SingleSiteRandomWalk(step_size=1.0)

    def test_beta_binomial_conjugate_run(self):
        mh = bm.SingleSiteRandomWalk(step_size=0.3)
        self.beta_binomial_conjugate_run(mh, num_samples=2000, delta=0.5)

    def test_gamma_gamma_conjugate_run(self):
        self.gamma_gamma_conjugate_run(self.mh, num_samples=2000, delta=0.2)

    def test_gamma_normal_conjugate_run(self):
        self.gamma_normal_conjugate_run(self.mh, num_samples=2000, delta=0.5)

    def test_normal_normal_conjugate_run(self):
        mh = bm.SingleSiteRandomWalk(step_size=1.5)
        self.normal_normal_conjugate_run(mh, num_samples=1000, delta=0.3)

    def test_distant_normal_normal_conjugate_run(self):
        mh = bm.SingleSiteRandomWalk(step_size=3.0)
        self.normal_normal_conjugate_run(mh, num_samples=5000, delta=1.0)

    def test_dirichlet_categorical_conjugate_run(self):
        self.dirichlet_categorical_conjugate_run(self.mh, num_samples=2000, delta=0.2)
