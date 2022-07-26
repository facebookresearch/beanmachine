#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Dirichlet compiler tests

import unittest

import beanmachine.ppl as bm
import torch
from beanmachine.ppl.inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli, Normal


@bm.random_variable
def norm(x):
    return Normal(0.0, 1.0)


@bm.random_variable
def bern():
    return Bernoulli(0.5)


@bm.functional
def cholesky1():
    n0 = norm(0) * norm(0)
    n1 = norm(1) * norm(1)
    t = tensor([[n0, 0.0], [0.0, n1]])
    return torch.linalg.cholesky(t)


@bm.functional
def cholesky2():
    n0 = norm(0) * norm(0)
    n1 = norm(1) * norm(1)
    t = tensor([[n0, 0.0], [0.0, n1]])
    return torch.Tensor.cholesky(t)


@bm.functional
def cholesky3():
    n0 = norm(0) * norm(0)
    n1 = norm(1) * norm(1)
    t = tensor([[n0, 0.0], [0.0, n1]])
    return t.cholesky()


@bm.functional
def cholesky4():
    # Matrix of bools should convert to reals
    t = tensor([[bern(), 0], [0, 1]])
    return t.cholesky()


@bm.functional
def cholesky5():
    n0 = norm(0) * norm(0)
    n1 = norm(1) * norm(1)
    t = tensor([[n0, 0.0], [0.0, n1]])
    L, _ = torch.linalg.cholesky_ex(t)
    return L


# TODO: Test with a non-square matrix, should give an error.


class CholeskyTest(unittest.TestCase):
    def test_cholesky(self) -> None:
        self.maxDiff = None

        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=Sample];
  N05[label=2];
  N06[label="*"];
  N07[label="*"];
  N08[label=ToMatrix];
  N09[label=Cholesky];
  N10[label=Query];
  N00 -> N02;
  N00 -> N08;
  N00 -> N08;
  N01 -> N02;
  N02 -> N03;
  N02 -> N04;
  N03 -> N06;
  N03 -> N06;
  N04 -> N07;
  N04 -> N07;
  N05 -> N08;
  N05 -> N08;
  N06 -> N08;
  N07 -> N08;
  N08 -> N09;
  N09 -> N10;
}
"""

        observed = BMGInference().to_dot([cholesky1()], {})
        self.assertEqual(expected.strip(), observed.strip())
        observed = BMGInference().to_dot([cholesky2()], {})
        self.assertEqual(expected.strip(), observed.strip())
        observed = BMGInference().to_dot([cholesky3()], {})
        self.assertEqual(expected.strip(), observed.strip())
        observed = BMGInference().to_dot([cholesky5()], {})
        self.assertEqual(expected.strip(), observed.strip())

        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=2];
  N4[label=False];
  N5[label=True];
  N6[label=ToMatrix];
  N7[label=ToRealMatrix];
  N8[label=Cholesky];
  N9[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N6;
  N3 -> N6;
  N3 -> N6;
  N4 -> N6;
  N4 -> N6;
  N5 -> N6;
  N6 -> N7;
  N7 -> N8;
  N8 -> N9;
}
        """
        observed = BMGInference().to_dot([cholesky4()], {})
        self.assertEqual(expected.strip(), observed.strip())
