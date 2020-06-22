# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.tests.conjugate_tests.abstract_conjugate import (
    AbstractConjugateTests,
)


class SingleSiteAdaptiveHamiltonianMonteCarloConjugateTest(
    unittest.TestCase, AbstractConjugateTests
):
    # for some transformed spaces, the acceptance ratio is never .65
    # see Adapative HMC Conjugate Tests for more details

    def test_beta_binomial_conjugate_run(self):
        pass

    def test_gamma_gamma_conjugate_run(self):
        pass

    def test_gamma_normal_conjugate_run(self):
        pass

    def test_normal_normal_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(1.0)
        self.normal_normal_conjugate_run(hmc, num_samples=300, delta=0.2)

    def test_distant_normal_normal_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(1.0)
        self.distant_normal_normal_conjugate_run(hmc, num_samples=500, delta=0.15)

    def test_dirichlet_categorical_conjugate_run(self):
        hmc = bm.SingleSiteHamiltonianMonteCarlo(0.1)
        self.dirichlet_categorical_conjugate_run(hmc, num_samples=200, delta=0.15)
