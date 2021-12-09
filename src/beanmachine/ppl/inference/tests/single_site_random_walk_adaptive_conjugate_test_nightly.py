# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.testlib.abstract_conjugate import AbstractConjugateTests


class SingleSiteAdaptiveRandomWalkConjugateTest(
    unittest.TestCase, AbstractConjugateTests
):
    def setUp(self):
        self.mh = bm.SingleSiteRandomWalk(step_size=5.0)

    def test_beta_binomial_conjugate_run(self):
        self.mh = bm.SingleSiteRandomWalk(step_size=1.0)
        self.beta_binomial_conjugate_run(
            self.mh, num_samples=3000, num_adaptive_samples=1600
        )

    @unittest.skip("Known to fail. Investigating in T77865889.")
    def test_gamma_gamma_conjugate_run(self):
        self.mh = bm.SingleSiteRandomWalk(step_size=3.0)
        self.gamma_gamma_conjugate_run(
            self.mh, num_samples=5000, num_adaptive_samples=7000
        )

    def test_gamma_normal_conjugate_run(self):
        self.mh = bm.SingleSiteRandomWalk(step_size=5.0)
        self.gamma_normal_conjugate_run(
            self.mh, num_samples=6000, num_adaptive_samples=5000
        )

    @unittest.skip("Known to fail. Investigating in T77865889.")
    def test_normal_normal_conjugate_run(self):
        self.normal_normal_conjugate_run(
            self.mh, num_samples=2000, num_adaptive_samples=2000
        )

    @unittest.skip("Known to fail. Investigating in T77865889.")
    def test_dirichlet_categorical_conjugate_run(self):
        self.dirichlet_categorical_conjugate_run(
            self.mh, num_samples=2000, num_adaptive_samples=2000
        )
