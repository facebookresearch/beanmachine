# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch.distributions import Bernoulli


# The dependency graph here is x -> y -> z -> x


@bm.random_variable
def bad_cycle_1_x():
    return Bernoulli(bad_cycle_1_y())


@bm.random_variable
def bad_cycle_1_y():
    return Bernoulli(bad_cycle_1_z())


@bm.random_variable
def bad_cycle_1_z():
    return Bernoulli(bad_cycle_1_x())


# The dependency graph here is z -> x(2) -> y(0) -> x(1) -> y(0)


@bm.random_variable
def bad_cycle_2_x(n):
    return Bernoulli(bad_cycle_2_y(0))


@bm.random_variable
def bad_cycle_2_y(n):
    return Bernoulli(bad_cycle_2_x(n + 1))


@bm.random_variable
def bad_cycle_2_z():
    return Bernoulli(bad_cycle_2_x(2))


class CycleDetectorTest(unittest.TestCase):
    def test_bad_cyclic_model_1(self) -> None:
        with self.assertRaises(RecursionError):
            BMGInference().infer([bad_cycle_1_x()], {}, 1)

    def test_bad_cyclic_model_2(self) -> None:
        with self.assertRaises(RecursionError):
            BMGInference().infer([bad_cycle_2_z()], {}, 1)
