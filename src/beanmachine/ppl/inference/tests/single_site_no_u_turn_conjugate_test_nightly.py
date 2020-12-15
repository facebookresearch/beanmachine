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
            nuts, num_samples=100, delta=0.05, num_adaptive_samples=50
        )
        nuts = SingleSiteNoUTurnSampler()
        self.beta_binomial_conjugate_run(
            nuts, num_samples=100, delta=0.05, num_adaptive_samples=50
        )

    def test_gamma_gamma_conjugate_run(self):
        nuts = SingleSiteNoUTurnSampler(use_dense_mass_matrix=False)
        self.gamma_gamma_conjugate_run(
            nuts, num_samples=200, delta=0.07, num_adaptive_samples=100
        )
        nuts = SingleSiteNoUTurnSampler()
        self.gamma_gamma_conjugate_run(
            nuts, num_samples=200, delta=0.07, num_adaptive_samples=100
        )

    def test_gamma_normal_conjugate_run(self):
        nuts = SingleSiteNoUTurnSampler(use_dense_mass_matrix=False)
        # TODO: The delta in the following needs to be reduced
        self.gamma_normal_conjugate_run(
            nuts, num_samples=150, delta=0.21, num_adaptive_samples=50
        )
        nuts = SingleSiteNoUTurnSampler()
        self.gamma_normal_conjugate_run(
            nuts, num_samples=150, delta=0.21, num_adaptive_samples=50
        )

    def test_normal_normal_conjugate_run(self):
        nuts = SingleSiteNoUTurnSampler(use_dense_mass_matrix=False)
        self.normal_normal_conjugate_run(
            nuts, num_samples=200, delta=0.08, num_adaptive_samples=100
        )
        nuts = SingleSiteNoUTurnSampler()
        self.normal_normal_conjugate_run(
            nuts, num_samples=500, delta=0.2, num_adaptive_samples=200
        )

    def test_distant_normal_normal_conjugate_run(self):
        nuts = SingleSiteNoUTurnSampler(use_dense_mass_matrix=False)
        self.distant_normal_normal_conjugate_run(
            nuts, num_samples=200, delta=0.15, num_adaptive_samples=100
        )
        nuts = SingleSiteNoUTurnSampler()
        self.distant_normal_normal_conjugate_run(
            nuts, num_samples=1000, delta=0.22, num_adaptive_samples=500
        )

    def test_dirichlet_categorical_conjugate_run(self):
        nuts = SingleSiteNoUTurnSampler(use_dense_mass_matrix=False)
        self.dirichlet_categorical_conjugate_run(
            nuts, num_samples=200, delta=0.05, num_adaptive_samples=100
        )
        nuts = SingleSiteNoUTurnSampler()
        self.dirichlet_categorical_conjugate_run(
            nuts, num_samples=200, delta=0.15, num_adaptive_samples=100
        )
