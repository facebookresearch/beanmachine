# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.testlib.abstract_conjugate import AbstractConjugateTests


class SingleSiteRandomWalkConjugateTest(unittest.TestCase, AbstractConjugateTests):
    def setUp(self):
        self.mh = bm.SingleSiteRandomWalk(step_size=1.0)
        self.seeds = {123456789, 975318642, 918273645, 564738291}

    def test_beta_binomial_conjugate_run(self):
        mh = bm.SingleSiteRandomWalk(step_size=0.3)
        for seed in self.seeds:
            self.beta_binomial_conjugate_run(mh, num_samples=5000, random_seed=seed)

    def test_gamma_gamma_conjugate_run(self):
        for seed in self.seeds:
            self.gamma_gamma_conjugate_run(self.mh, num_samples=10000, random_seed=seed)

    def test_gamma_normal_conjugate_run(self):
        for seed in self.seeds:
            self.gamma_normal_conjugate_run(
                self.mh, num_samples=10000, random_seed=seed
            )

    def test_normal_normal_conjugate_run(self):
        mh = bm.SingleSiteRandomWalk(step_size=1.5)
        for seed in self.seeds:
            self.normal_normal_conjugate_run(mh, num_samples=1000, random_seed=seed)

    def test_distant_normal_normal_conjugate_run(self):
        mh = bm.SingleSiteRandomWalk(step_size=3.0)
        for seed in self.seeds:
            self.normal_normal_conjugate_run(mh, num_samples=10000, random_seed=seed)

    def test_dirichlet_categorical_conjugate_run(self):
        for seed in self.seeds:
            self.dirichlet_categorical_conjugate_run(
                self.mh, num_samples=10000, random_seed=seed
            )
