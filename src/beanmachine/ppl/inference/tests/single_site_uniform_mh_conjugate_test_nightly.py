# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from beanmachine.ppl.experimental.global_inference.single_site_uniform_mh import (
    SingleSiteUniformMetropolisHastings,
)
from beanmachine.ppl.testlib.abstract_conjugate import AbstractConjugateTests


class SingleSiteUniformMetropolisHastingsConjugateTest(
    unittest.TestCase, AbstractConjugateTests
):
    def setUp(self):
        self.mh = SingleSiteUniformMetropolisHastings()

    def test_beta_binomial_conjugate_run(self):
        self.beta_binomial_conjugate_run(self.mh)

    def test_gamma_gamma_conjugate_run(self):
        self.gamma_gamma_conjugate_run(self.mh)

    def test_gamma_normal_conjugate_run(self):
        self.gamma_normal_conjugate_run(self.mh, num_samples=7500)

    def test_normal_normal_conjugate_run(self):
        self.normal_normal_conjugate_run(self.mh, num_samples=5000)

    def test_dirichlet_categorical_conjugate_run(self):
        self.dirichlet_categorical_conjugate_run(self.mh, num_samples=5000)
