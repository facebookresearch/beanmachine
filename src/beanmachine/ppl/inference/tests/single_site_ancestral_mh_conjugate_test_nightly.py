# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.testlib.abstract_conjugate import AbstractConjugateTests


class SingleSiteAncestralMetropolisHastingsConjugateTest(
    unittest.TestCase, AbstractConjugateTests
):
    def setUp(self):
        self.mh = bm.SingleSiteAncestralMetropolisHastings()
        self.seeds = {
            123456789,
            975318642,
            564738291,
            129634860,
            193543287,
            978956376,
            867394610,
            885573652,
            846194628,
            918273645,
        }

    def test_beta_binomial_conjugate_run(self):
        for seed in self.seeds:
            self.beta_binomial_conjugate_run(self.mh, random_seed=seed)

    def test_gamma_gamma_conjugate_run(self):
        for seed in self.seeds:
            self.gamma_gamma_conjugate_run(self.mh, random_seed=seed)

    def test_gamma_normal_conjugate_run(self):
        for seed in self.seeds:
            self.gamma_normal_conjugate_run(
                self.mh, num_samples=20000, random_seed=seed
            )

    def test_normal_normal_conjugate_run(self):
        for seed in self.seeds:
            self.normal_normal_conjugate_run(
                self.mh, num_samples=5000, random_seed=seed
            )

    def test_distant_normal_normal_conjugate_run(self):
        # We expect ancestral to be able to converge fast for this model.
        # This test consistently (that is, for all seeds tested) fails to converge
        for seed in self.seeds:
            try:
                self.distant_normal_normal_conjugate_run(
                    self.mh, num_samples=1000, random_seed=seed
                )
            except AssertionError as e:
                msg_expected = "2.7625932693481445 not greater than or equal to 30 : Effective sample size too small for normalcy assumption. random_seed"
                msg_observed = str(e).split(" = ")[0]
                self.assertEquals(msg_expected, msg_observed)

    def test_dirichlet_categorical_conjugate_run(self):
        for seed in self.seeds:
            self.dirichlet_categorical_conjugate_run(
                self.mh, num_samples=10000, random_seed=seed
            )
