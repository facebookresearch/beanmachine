# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
from ..testlib.abstract_conjugate import AbstractConjugateTests


class SingleSiteRandomWalkConjugateTest(unittest.TestCase, AbstractConjugateTests):
    def setUp(self):
        self.mh = bm.SingleSiteRandomWalk(step_size=1.0)

    def test_beta_binomial_conjugate_run(self):
        mh = bm.SingleSiteRandomWalk(step_size=0.3)
        self.beta_binomial_conjugate_run(mh, num_samples=5000)

    def test_gamma_gamma_conjugate_run(self):
        self.gamma_gamma_conjugate_run(self.mh, num_samples=10000)

    def test_gamma_normal_conjugate_run(self):
        self.gamma_normal_conjugate_run(self.mh, num_samples=10000)

    def test_normal_normal_conjugate_run(self):
        mh = bm.SingleSiteRandomWalk(step_size=1.5)
        self.normal_normal_conjugate_run(mh, num_samples=1000)

    def test_dirichlet_categorical_conjugate_run(self):
        self.dirichlet_categorical_conjugate_run(self.mh, num_samples=10000)
