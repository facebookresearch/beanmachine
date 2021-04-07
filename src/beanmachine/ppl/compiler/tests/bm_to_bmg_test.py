# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bm_to_bmg.py"""
import unittest

import astor
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bm_to_bmg import (
    _bm_function_to_bmg_function,
    to_bmg,
    to_cpp,
    to_dot,
    to_python,
    to_python_raw,
)


# flake8 does not provide any mechanism to disable warnings in
# multi-line strings, so just turn it off for this file.
# flake8: noqa


def tidy(s: str) -> str:
    return "\n".join(c.strip() for c in s.strip().split("\n")).strip()


# Bayesian regression
source10 = """
import beanmachine.ppl as bm
from torch import tensor, zeros
from torch.distributions import Normal, Bernoulli
N = 3
K = 2
X = tensor([[1.0, 10, 20], [1.0, -100, -190], [1.0, -101, -192]])
intercept_scale = 0.9
coef_scale = [1.2, 2.3]

@bm.random_variable
def beta():
    return Normal(
        zeros((K + 1, 1)), tensor([intercept_scale] + coef_scale).view(K + 1, 1)
    )

@bm.random_variable
def y():
    return Bernoulli(logits=X.mm(beta()))
"""

expected_raw_10 = """
from beanmachine.ppl.utils.memoize import memoize
from beanmachine.ppl.utils.probabilistic import probabilistic
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
_lifted_to_bmg: bool = True
bmg = BMGraphBuilder()
from torch import tensor, zeros
from torch.distributions import Normal, Bernoulli
N = 3
K = 2
a9 = 1.0
a14 = 10
a19 = 20
a5 = [a9, a14, a19]
a15 = 1.0
a20 = -100
a24 = -190
a10 = [a15, a20, a24]
a21 = 1.0
a25 = -101
a29 = -192
a16 = [a21, a25, a29]
a1 = [a5, a10, a16]
X = bmg.handle_function(tensor, [a1], {})
intercept_scale = 0.9
a2 = 1.2
a6 = 2.3
coef_scale = [a2, a6]


@probabilistic(bmg)
@memoize
def beta():
    a11 = K + 1, 1
    a7 = bmg.handle_function(zeros, [a11], {})
    a30 = [intercept_scale]
    a26 = bmg.handle_addition(a30, coef_scale)
    a22 = bmg.handle_function(tensor, [a26], {})
    a17 = bmg.handle_dot_get(a22, 'view')
    a27 = 1
    a23 = bmg.handle_addition(K, a27)
    a28 = 1
    a12 = bmg.handle_function(a17, [a23, a28], {})
    r3 = bmg.handle_function(Normal, [a7, a12], {})
    return bmg.handle_sample(r3)


@probabilistic(bmg)
@memoize
def y():
    a13 = bmg.handle_dot_get(X, 'mm')
    a18 = bmg.handle_function(beta, [], {})
    a8 = bmg.handle_function(a13, [a18], {})
    r4 = bmg.handle_function(Bernoulli, [], {**{'logits': a8}})
    return bmg.handle_sample(r4)


roots = [beta(), y()]
"""

expected_dot_10 = """
digraph "graph" {
  N0[label="[[0.0],\\\\n[0.0],\\\\n[0.0]]"];
  N1[label="[[0.8999999761581421],\\\\n[1.2000000476837158],\\\\n[2.299999952316284]]"];
  N2[label=Normal];
  N3[label=Sample];
  N4[label="[[1.0,10.0,20.0],\\\\n[1.0,-100.0,-190.0],\\\\n[1.0,-101.0,-192.0]]"];
  N5[label="*"];
  N6[label="Bernoulli(logits)"];
  N7[label=Sample];
  N2 -> N0[label=mu];
  N2 -> N1[label=sigma];
  N3 -> N2[label=operand];
  N5 -> N3[label=right];
  N5 -> N4[label=left];
  N6 -> N5[label=probability];
  N7 -> N6[label=operand];
}
"""

# A sketch of a model for predicting if a new account is fake based on
# friend requests issued and accepted.
source11 = """
import beanmachine.ppl as bm
from torch import tensor
from torch.distributions import Bernoulli
FAKE_PRIOR = 0.001
# One entry per user
FAKE_REQ_PROB = tensor([0.01, 0.02, 0.03])
REAL_REQ_PROB = tensor([0.04, 0.05, 0.06])
REQ_PROB = [REAL_REQ_PROB, FAKE_REQ_PROB]
REAL_ACC_PROB = tensor([0.99, 0.50, 0.07])
@bm.random_variable
def is_fake(account):
  return Bernoulli(FAKE_PRIOR)
@bm.random_variable
def all_requests_sent(account):
  return Bernoulli(REQ_PROB[is_fake(account)])
@bm.random_variable
def all_requests_accepted(account):
  return Bernoulli(REAL_ACC_PROB * all_requests_sent(account))
_1 = 0
_2 = all_requests_accepted(_1)
"""

expected_raw_11 = """
import beanmachine.ppl as bm
from beanmachine.ppl.utils.memoize import memoize
from beanmachine.ppl.utils.probabilistic import probabilistic
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
_lifted_to_bmg: bool = True
bmg = BMGraphBuilder()
from torch import tensor
from torch.distributions import Bernoulli
FAKE_PRIOR = 0.001
a14 = 0.01
a19 = 0.02
a24 = 0.03
a8 = [a14, a19, a24]
r4 = [a8]
FAKE_REQ_PROB = bmg.handle_function(tensor, [*r4], {})
a15 = 0.04
a20 = 0.05
a25 = 0.06
a9 = [a15, a20, a25]
r5 = [a9]
REAL_REQ_PROB = bmg.handle_function(tensor, [*r5], {})
REQ_PROB = [REAL_REQ_PROB, FAKE_REQ_PROB]
a16 = 0.99
a21 = 0.5
a26 = 0.07
a10 = [a16, a21, a26]
r6 = [a10]
REAL_ACC_PROB = bmg.handle_function(tensor, [*r6], {})


@probabilistic(bmg)
@memoize
def is_fake(account):
    r11 = [FAKE_PRIOR]
    r1 = bmg.handle_function(Bernoulli, [*r11], {})
    return bmg.handle_sample(r1)


@probabilistic(bmg)
@memoize
def all_requests_sent(account):
    r27 = [account]
    a22 = bmg.handle_function(is_fake, [*r27], {})
    a17 = bmg.handle_index(REQ_PROB, a22)
    r12 = [a17]
    r2 = bmg.handle_function(Bernoulli, [*r12], {})
    return bmg.handle_sample(r2)


@probabilistic(bmg)
@memoize
def all_requests_accepted(account):
    r28 = [account]
    a23 = bmg.handle_function(all_requests_sent, [*r28], {})
    a18 = bmg.handle_multiplication(REAL_ACC_PROB, a23)
    r13 = [a18]
    r3 = bmg.handle_function(Bernoulli, [*r13], {})
    return bmg.handle_sample(r3)


_1 = 0
r7 = [_1]
_2 = bmg.handle_function(all_requests_accepted, [*r7], {})
roots = []
"""

expected_dot_11 = """
digraph "graph" {
  N00[label=0.0010000000474974513];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=0];
  N04[label="[0.03999999910593033,0.05000000074505806,0.05999999865889549]"];
  N05[label=1];
  N06[label="[0.009999999776482582,0.019999999552965164,0.029999999329447746]"];
  N07[label=map];
  N08[label=index];
  N09[label=Bernoulli];
  N10[label=Sample];
  N11[label="[0.9900000095367432,0.5,0.07000000029802322]"];
  N12[label="*"];
  N13[label=Bernoulli];
  N14[label=Sample];
  N01 -> N00[label=probability];
  N02 -> N01[label=operand];
  N07 -> N03[label=0];
  N07 -> N04[label=1];
  N07 -> N05[label=2];
  N07 -> N06[label=3];
  N08 -> N02[label=right];
  N08 -> N07[label=left];
  N09 -> N08[label=probability];
  N10 -> N09[label=operand];
  N12 -> N10[label=right];
  N12 -> N11[label=left];
  N13 -> N12[label=probability];
  N14 -> N13[label=operand];
}
"""


# Illustrate that we correctly generate the support for
# multidimensional Bernoulli distributions. Flip two coins,
# take their average, and use that to make a third coin:
source13 = """
import beanmachine.ppl as bm
import torch
from torch import tensor
from torch.distributions import Bernoulli

@bm.random_variable
def x(n):
  return Bernoulli(n.sum()*0.5)

@bm.random_variable
def y():
  return Bernoulli(tensor([0.5,0.5]))

@bm.random_variable
def z():
  return Bernoulli(x(y()))
"""

expected_dot_13 = """
digraph "graph" {
  N00[label="[0.5,0.5]"];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=0.0];
  N04[label=Bernoulli];
  N05[label=Sample];
  N06[label="[0.0,0.0]"];
  N07[label=0.5];
  N08[label=Bernoulli];
  N09[label=Sample];
  N10[label="[0.0,1.0]"];
  N11[label=Sample];
  N12[label="[1.0,0.0]"];
  N13[label=1.0];
  N14[label=Bernoulli];
  N15[label=Sample];
  N16[label="[1.0,1.0]"];
  N17[label=map];
  N18[label=index];
  N19[label=Bernoulli];
  N20[label=Sample];
  N01 -> N00[label=probability];
  N02 -> N01[label=operand];
  N04 -> N03[label=probability];
  N05 -> N04[label=operand];
  N08 -> N07[label=probability];
  N09 -> N08[label=operand];
  N11 -> N08[label=operand];
  N14 -> N13[label=probability];
  N15 -> N14[label=operand];
  N17 -> N05[label=1];
  N17 -> N06[label=0];
  N17 -> N09[label=3];
  N17 -> N10[label=2];
  N17 -> N11[label=5];
  N17 -> N12[label=4];
  N17 -> N15[label=7];
  N17 -> N16[label=6];
  N18 -> N02[label=right];
  N18 -> N17[label=left];
  N19 -> N18[label=probability];
  N20 -> N19[label=operand];
}
"""

# Simple example of categorical
source14 = """
import beanmachine.ppl as bm
import torch
from torch.distributions import Bernoulli, Categorical
from torch import tensor

@bm.random_variable
def x(n):
  if n == 0:
    return Bernoulli(0.5)
  if n == 1:
    return Categorical(tensor([1.0, 3.0, 4.0]))
  return Bernoulli(0.75)

@bm.random_variable
def y():
  return Categorical(tensor([2.0, 6.0, 8.0]))

@bm.random_variable
def z():
  p = x(y()) * 0.25
  return Bernoulli(p)
"""

expected_dot_14 = """
digraph "graph" {
  N00[label="[0.125,0.375,0.5]"];
  N01[label=Categorical];
  N02[label=Sample];
  N03[label=0.5];
  N04[label=Bernoulli];
  N05[label=Sample];
  N06[label=0];
  N07[label=Sample];
  N08[label=1];
  N09[label=0.75];
  N10[label=Bernoulli];
  N11[label=Sample];
  N12[label=2];
  N13[label=map];
  N14[label=index];
  N15[label=0.25];
  N16[label="*"];
  N17[label=Bernoulli];
  N18[label=Sample];
  N01 -> N00[label=probability];
  N02 -> N01[label=operand];
  N04 -> N03[label=probability];
  N05 -> N04[label=operand];
  N07 -> N01[label=operand];
  N10 -> N09[label=probability];
  N11 -> N10[label=operand];
  N13 -> N05[label=1];
  N13 -> N06[label=0];
  N13 -> N07[label=3];
  N13 -> N08[label=2];
  N13 -> N11[label=5];
  N13 -> N12[label=4];
  N14 -> N02[label=right];
  N14 -> N13[label=left];
  N16 -> N14[label=left];
  N16 -> N15[label=right];
  N17 -> N16[label=probability];
  N18 -> N17[label=operand];
}
"""


class CompilerTest(unittest.TestCase):
    def disabled_test_to_python_raw_10(self) -> None:
        # TODO: Enable this test when we support compilation of
        # TODO: vectorized models.
        self.maxDiff = None
        observed = to_python_raw(source10)
        self.assertEqual(observed.strip(), expected_raw_10.strip())

    def disabled_test_to_python_raw_11(self) -> None:
        # TODO: Enable this test when we support compilation of
        # TODO: vectorized models.
        self.maxDiff = None
        observed = to_python_raw(source11)
        self.assertEqual(observed.strip(), expected_raw_11.strip())

    def disabled_test_to_dot_10(self) -> None:
        # TODO: This crashes; something is broken with matrix multiplication.
        self.maxDiff = None
        observed = to_dot(source10)
        self.assertEqual(observed.strip(), expected_dot_10.strip())

    def disabled_test_to_dot_11(self) -> None:
        # TODO: This test is disabled because we do not yet support the indexing
        # operation where we choose via a stochastic list index which tensor to use.
        self.maxDiff = None
        observed = to_dot(source11)
        self.assertEqual(observed.strip(), expected_dot_11.strip())

    def test_to_dot_13(self) -> None:
        self.maxDiff = None
        observed = to_dot(source13)
        self.assertEqual(observed.strip(), expected_dot_13.strip())

    def test_to_dot_14(self) -> None:
        self.maxDiff = None
        observed = to_dot(source14)
        self.assertEqual(observed.strip(), expected_dot_14.strip())
