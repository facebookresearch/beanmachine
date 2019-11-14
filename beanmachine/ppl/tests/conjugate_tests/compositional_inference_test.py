# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from beanmachine.ppl.inference.single_site_compositional_infer import (
    SingleSiteCompositionalInference,
)
from beanmachine.ppl.tests.conjugate_tests.abstract_conjugate import (
    AbstractConjugateTests,
)


class SingleSiteCompositionalInferenceConjugateTest(
    unittest.TestCase, AbstractConjugateTests
):
    def setUp(self):
        self.mh = SingleSiteCompositionalInference()

    def test_beta_binomial_conjugate_run(self):
        self.beta_binomial_conjugate_run(self.mh)

    def test_gamma_gamma_conjugate_run(self):
        self.gamma_gamma_conjugate_run(self.mh)

    def test_gamma_normal_conjugate_run(self):
        self.gamma_normal_conjugate_run(self.mh, delta=0.2)

    def test_normal_normal_conjugate_run(self):
        self.normal_normal_conjugate_run(self.mh, delta=0.1)

    def test_distant_normal_normal_conjugate_run(self):
        self.distant_normal_normal_conjugate_run(self.mh, delta=0.1)

    def test_dirichlet_categorical_conjugate_run(self):
        self.dirichlet_categorical_conjugate_run(self.mh, delta=0.1)
