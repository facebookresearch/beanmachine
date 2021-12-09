# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch.distributions import Normal, StudentT


# Comparisons involving a graph node have no representation in
# BMG and should produce an error.
#
# Tests which show how the comparison operators are lowered into
# a more fundamental form are in comparison_rewriting_test.py.


@bm.random_variable
def normal():
    return Normal(0.0, 1.0)


@bm.random_variable
def t():
    gt = normal() > 2.0
    gte = normal() >= 2.0
    lt = normal() < 2.0
    lte = normal() <= 2.0
    _is = normal() is normal()
    is_not = normal() is not normal()
    _in = normal() in normal()
    not_in = normal() not in normal()
    s = 1.0 + gt + gte + lt + lte + _is + is_not + _in + not_in
    return StudentT(1.0, s, 1.0)


class ComparisonErrorsTest(unittest.TestCase):
    def test_comparison_errors_1(self) -> None:

        self.maxDiff = None
        bmg = BMGInference()

        # TODO: Raise a better error than a generic ValueError
        with self.assertRaises(ValueError) as ex:
            bmg.infer([t()], {}, 10)
        expected = """
The model uses a < operation unsupported by Bean Machine Graph.
The unsupported node is the right of a +.
The model uses a <= operation unsupported by Bean Machine Graph.
The unsupported node is the right of a +.
The model uses a > operation unsupported by Bean Machine Graph.
The unsupported node is the right of a +.
The model uses a >= operation unsupported by Bean Machine Graph.
The unsupported node is the right of a +.
The model uses a In operation unsupported by Bean Machine Graph.
The unsupported node is the right of a +.
The model uses a Is operation unsupported by Bean Machine Graph.
The unsupported node is the right of a +.
The model uses a IsNot operation unsupported by Bean Machine Graph.
The unsupported node is the right of a +.
The model uses a NotIn operation unsupported by Bean Machine Graph.
The unsupported node is the right of a +.
"""
        observed = str(ex.exception)
        self.assertEqual(expected.strip(), observed.strip())
