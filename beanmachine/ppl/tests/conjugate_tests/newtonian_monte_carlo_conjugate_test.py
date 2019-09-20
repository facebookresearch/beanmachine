# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from beanmachine.ppl.inference.single_site_newtonian_monte_carlo import (
    SingleSiteNewtonianMonteCarlo,
)
from beanmachine.ppl.tests.conjugate_tests.abstract_conjugate import (
    AbstractConjugateTests,
)


class SingleSiteNewtonianMonteCarloConjugateTest(
    unittest.TestCase, AbstractConjugateTests
):
    def test_beta_binomial_conjugate_run(self):
        pass

    def test_gamma_gamma_conjugate_run(self):
        pass

    def test_gamma_normal_conjugate_run(self):
        pass

    def test_normal_normal_conjugate_run(self):
        # Converges with 10k and more iterations but will use a bigger delta for
        # now to have a faster test.
        expected_mean, expected_std, queries, observations = (
            self.compute_normal_normal_moments()
        )
        nw = SingleSiteNewtonianMonteCarlo()
        predictions = nw.infer(queries, observations, 500)
        mean, _ = self.compute_statistics(predictions[queries[0]])
        self.assertAlmostEqual(abs((mean - expected_mean).sum().item()), 0, delta=0.15)

    def test_distant_normal_normal_conjugate_run(self):
        # Converges with 10k and more iterations but will use a bigger delta for
        # now to have a faster test.
        expected_mean, expected_std, queries, observations = (
            self.compute_distant_normal_normal_moments()
        )
        nw = SingleSiteNewtonianMonteCarlo()
        predictions = nw.infer(queries, observations, 800)
        mean, _ = self.compute_statistics(predictions[queries[0]])
        self.assertAlmostEqual(abs((mean - expected_mean).sum().item()), 0, delta=0.15)

    def test_dirichlet_categorical_conjugate_run(self):
        pass
