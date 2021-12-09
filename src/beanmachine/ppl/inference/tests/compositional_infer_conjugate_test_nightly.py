# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from beanmachine.ppl.inference.compositional_infer import (
    CompositionalInference,
)
from beanmachine.ppl.testlib.abstract_conjugate import AbstractConjugateTests


class CompositionalInferenceConjugateTest(unittest.TestCase, AbstractConjugateTests):
    def setUp(self):
        self.mh = CompositionalInference()

    def test_beta_binomial_conjugate_run(self):
        self.beta_binomial_conjugate_run(self.mh)

    def test_gamma_gamma_conjugate_run(self):
        self.gamma_gamma_conjugate_run(self.mh)

    def test_gamma_normal_conjugate_run(self):
        self.gamma_normal_conjugate_run(self.mh)

    def test_normal_normal_conjugate_run(self):
        self.normal_normal_conjugate_run(self.mh)

    def test_dirichlet_categorical_conjugate_run(self):
        self.dirichlet_categorical_conjugate_run(self.mh)
