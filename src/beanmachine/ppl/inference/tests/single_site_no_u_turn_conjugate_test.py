# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from beanmachine.ppl.inference.single_site_no_u_turn_sampler import (
    SingleSiteNoUTurnSampler,
)
from beanmachine.ppl.testlib.abstract_conjugate import AbstractConjugateTests


class SingleSiteNoUTurnConjugateTest(unittest.TestCase, AbstractConjugateTests):
    # for some transformed spaces, the acceptance ratio is never .65
    # see Adapative HMC Conjugate Tests for more details

    def test_beta_binomial_conjugate_run(self):
        nuts = SingleSiteNoUTurnSampler(use_dense_mass_matrix=False)
        self.beta_binomial_conjugate_run(
            nuts, num_samples=200, num_adaptive_samples=100
        )
        nuts = SingleSiteNoUTurnSampler()
        self.beta_binomial_conjugate_run(
            nuts, num_samples=200, num_adaptive_samples=100
        )

    def test_gamma_gamma_conjugate_run(self):
        nuts = SingleSiteNoUTurnSampler(use_dense_mass_matrix=False)
        self.gamma_gamma_conjugate_run(nuts, num_samples=500, num_adaptive_samples=300, random_seed=256)
        nuts = SingleSiteNoUTurnSampler()
        self.gamma_gamma_conjugate_run(nuts, num_samples=500, num_adaptive_samples=300, random_Seed=256)

    def test_gamma_normal_conjugate_run(self):
        nuts = SingleSiteNoUTurnSampler(use_dense_mass_matrix=False)
        self.gamma_normal_conjugate_run(nuts, num_samples=300, num_adaptive_samples=300)
        nuts = SingleSiteNoUTurnSampler()
        self.gamma_normal_conjugate_run(nuts, num_samples=300, num_adaptive_samples=300)

    def test_normal_normal_conjugate_run(self):
        nuts = SingleSiteNoUTurnSampler(use_dense_mass_matrix=False)
        self.normal_normal_conjugate_run(
            nuts, num_samples=200, num_adaptive_samples=100
        )
        # Note: Following test failed on mean eq at 500/250
        nuts = SingleSiteNoUTurnSampler()
        self.normal_normal_conjugate_run(
            nuts, num_samples=600, num_adaptive_samples=300
        )

    def test_dirichlet_categorical_conjugate_run(self):
        nuts = SingleSiteNoUTurnSampler(use_dense_mass_matrix=False)
        self.dirichlet_categorical_conjugate_run(
            nuts, num_samples=400, num_adaptive_samples=400
        )
        nuts = SingleSiteNoUTurnSampler()
        self.dirichlet_categorical_conjugate_run(
            nuts, num_samples=400, num_adaptive_samples=400
        )
