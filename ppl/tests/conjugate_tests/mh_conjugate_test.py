# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from beanmachine.ppl.inference.single_site_ancestral_mh import (
    SingleSiteAncestralMetropolisHastings,
)
from beanmachine.ppl.tests.conjugate_tests.abstract_conjugate import (
    AbstractConjugateTests,
)


class SingleSiteAncestralMetropolisHastingsConjugateTest(
    unittest.TestCase, AbstractConjugateTests
):
    def test_beta_binomial_conjugate_run(self):
        mean, expected_mean, std, expected_std = self.compute_beta_binomial_moments(
            1000, SingleSiteAncestralMetropolisHastings
        )
        self.assertAlmostEqual(abs((mean - expected_mean).sum().item()), 0, delta=0.05)

    def test_gamma_gamma_conjugate_run(self):
        mean, expected_mean, std, expected_std = self.compute_gamma_gamma_moments(
            1000, SingleSiteAncestralMetropolisHastings
        )

        self.assertAlmostEqual(abs((mean - expected_mean).sum().item()), 0, delta=0.05)

    def test_gamma_normal_conjugate_run(self):
        # Converges with 10k and more iterations but will use a bigger delta for
        # now to have a faster test.
        mean, expected_mean, std, expected_std = self.compute_gamma_normal_moments(
            7500, SingleSiteAncestralMetropolisHastings
        )
        self.assertAlmostEqual(abs((mean - expected_mean).sum().item()), 0, delta=0.1)

    def test_normal_normal_conjugate_run(self):
        # Converges with 10k and more iterations but will use a bigger delta for
        # now to have a faster test.
        mean, expected_mean, std, expected_std = self.compute_normal_normal_moments(
            5000, SingleSiteAncestralMetropolisHastings
        )
        self.assertAlmostEqual(abs((mean - expected_mean).sum().item()), 0, delta=0.1)

    def test_dirichlet_categorical_conjugate_run(self):
        mean, expected_mean, std, expected_std = self.compute_dirichlet_categorical_moments(
            5000, SingleSiteAncestralMetropolisHastings
        )
        self.assertAlmostEqual(abs((mean - expected_mean).sum().item()), 0, delta=0.1)
