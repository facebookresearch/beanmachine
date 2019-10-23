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
    def setUp(self):
        self.mh = SingleSiteNewtonianMonteCarlo()

    def test_beta_binomial_conjugate_run(self):
        pass

    def test_gamma_gamma_conjugate_run(self):
        pass

    def test_gamma_normal_conjugate_run(self):
        pass

    def test_normal_normal_conjugate_run(self):
        # Converges with 10k and more iterations but will use a bigger delta for
        # now to have a faster test.
        self.normal_normal_conjugate_run(self.mh, num_samples=500, delta=0.15)

    def test_distant_normal_normal_conjugate_run(self):
        # Converges with 10k and more iterations but will use a bigger delta for
        # now to have a faster test.
        self.distant_normal_normal_conjugate_run(self.mh, num_samples=800, delta=0.15)

    def test_dirichlet_categorical_conjugate_run(self):
        pass
