# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from beanmachine.ppl.inference.single_site_random_walk import SingleSiteRandomWalk
from beanmachine.ppl.tests.conjugate_tests.abstract_conjugate import (
    AbstractConjugateTests,
)


class SingleSiteRandomWalkConjugateTest(unittest.TestCase, AbstractConjugateTests):
    def test_beta_binomial_conjugate_run(self):
        expected_mean, expected_std, queries, observations = (
            self.compute_beta_binomial_moments()
        )
        mh = SingleSiteRandomWalk(step_size=2.0)
        predictions = mh.infer(queries, observations, 2000)
        mean, _ = self.compute_statistics(predictions.get_chain()[queries[0]])
        # Smaller delta, within reasonable time, would likely require a burn-in period
        self.assertAlmostEqual(abs((mean - expected_mean).sum().item()), 0, delta=0.5)

    def test_gamma_gamma_conjugate_run(self):
        expected_mean, expected_std, queries, observations = (
            self.compute_gamma_gamma_moments()
        )
        mh = SingleSiteRandomWalk(step_size=1.0)
        predictions = mh.infer(queries, observations, 2000)
        mean, _ = self.compute_statistics(predictions.get_chain()[queries[0]])
        self.assertAlmostEqual(abs((mean - expected_mean).sum().item()), 0, delta=0.2)

    def test_gamma_normal_conjugate_run(self):
        # Converges with 10k and more iterations but will use a bigger delta for
        # now to have a faster test.
        expected_mean, expected_std, queries, observations = (
            self.compute_gamma_normal_moments()
        )
        mh = SingleSiteRandomWalk(step_size=1.0)
        predictions = mh.infer(queries, observations, 2000)
        mean, _ = self.compute_statistics(predictions.get_chain()[queries[0]])
        self.assertAlmostEqual(abs((mean - expected_mean).sum().item()), 0, delta=0.5)

    def test_normal_normal_conjugate_run(self):
        # Converges with 10k and more iterations but will use a bigger delta for
        # now to have a faster test.
        expected_mean, expected_std, queries, observations = (
            self.compute_normal_normal_moments()
        )
        mh = SingleSiteRandomWalk()
        predictions = mh.infer(queries, observations, 1000)
        mean, _ = self.compute_statistics(predictions.get_chain()[queries[0]])
        self.assertAlmostEqual(abs((mean - expected_mean).sum().item()), 0, delta=0.3)

    def test_distant_normal_normal_conjugate_run(self):
        # Converges with 10k and more iterations but will use a bigger delta for
        # now to have a faster test.
        expected_mean, expected_std, queries, observations = (
            self.compute_distant_normal_normal_moments()
        )
        mh = SingleSiteRandomWalk(step_size=3.0)
        predictions = mh.infer(queries, observations, 5000)
        mean, _ = self.compute_statistics(predictions.get_chain()[queries[0]])
        # Smaller delta, within reasonable time, would likely require a burn-in period
        self.assertAlmostEqual(abs((mean - expected_mean).sum().item()), 0, delta=1.0)

    def test_dirichlet_categorical_conjugate_run(self):
        pass
