# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.testlib.abstract_conjugate import AbstractConjugateTests


class SingleSiteNoUTurnConjugateTest(unittest.TestCase, AbstractConjugateTests):
    def setUp(self):
        self.nuts = bm.SingleSiteNoUTurnSampler()

    def test_beta_binomial_conjugate_run(self):
        self.beta_binomial_conjugate_run(
            self.nuts, num_samples=300, num_adaptive_samples=300
        )

    def test_gamma_gamma_conjugate_run(self):
        self.gamma_gamma_conjugate_run(
            self.nuts, num_samples=300, num_adaptive_samples=300
        )

    def test_gamma_normal_conjugate_run(self):
        self.gamma_normal_conjugate_run(
            self.nuts, num_samples=300, num_adaptive_samples=300
        )

    def test_normal_normal_conjugate_run(self):
        self.normal_normal_conjugate_run(
            self.nuts, num_samples=300, num_adaptive_samples=300
        )

    def test_dirichlet_categorical_conjugate_run(self):
        self.dirichlet_categorical_conjugate_run(
            self.nuts, num_samples=300, num_adaptive_samples=300
        )
