# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.testlib.abstract_conjugate import AbstractConjugateTests


class CompositionalInferenceConjugateTest(unittest.TestCase, AbstractConjugateTests):
    def setUp(self):
        self.mh = bm.CompositionalInference()

    def test_beta_binomial_conjugate_run(self):
        self.beta_binomial_conjugate_run(self.mh)

    def test_gamma_gamma_conjugate_run(self):
        self.gamma_gamma_conjugate_run(self.mh)

    def test_gamma_normal_conjugate_run(self):
        self.gamma_normal_conjugate_run(self.mh, delta=0.2)

    def test_normal_normal_conjugate_run(self):
        self.normal_normal_conjugate_run(self.mh, delta=0.1)

    def test_distant_normal_normal_conjugate_run(self):
        self.distant_normal_normal_conjugate_run(self.mh, num_samples=1000, delta=0.1)

    def test_dirichlet_categorical_conjugate_run(self):
        self.dirichlet_categorical_conjugate_run(self.mh, delta=0.1)
