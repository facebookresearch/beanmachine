# Copyright (c) Facebook, Inc. and its affiliates.
import unittest
from typing import Any

import beanmachine.ppl as bm
from beanmachine.ppl.compiler.runtime import BMGRuntime
from beanmachine.ppl.compiler.support import ComputeSupport, Infinite, TooBig
from torch import Tensor, tensor
from torch.distributions import Bernoulli, Normal, Categorical


def tidy(s: str) -> str:
    return "\n".join(c.strip() for c in s.strip().split("\n")).strip()


def tensor_equality(x: Tensor, y: Tensor) -> bool:
    # Tensor equality is weird.  Suppose x and y are both
    # tensor([1.0, 2.0]). Then x.eq(y) is tensor([True, True]),
    # and x.eq(y).all() is tensor(True).
    return bool(x.eq(y).all())


@bm.random_variable
def flip1(n):
    return Bernoulli(0.5)


@bm.random_variable
def flip2(n):
    return Bernoulli(tensor([[0.5, 0.5]]))


@bm.functional
def to_tensor():
    return tensor([2.5, flip1(0), flip1(1), flip1(2)])


@bm.random_variable
def normal():
    return Normal(0.0, 1.0)


@bm.functional
def sum1():
    return flip1(0) + 1.0


@bm.functional
def prod1():
    return sum1() * sum1()


@bm.functional
def pow1():
    return prod1() ** prod1()


@bm.functional
def ge1():
    return pow1() >= prod1()


@bm.functional
def and1():
    return ge1() & ge1()


@bm.functional
def negexp1():
    return -prod1().exp()


@bm.random_variable
def cat3():
    return Categorical(tensor([0.5, 0.25, 0.25]))


@bm.random_variable
def cat2_3():
    return Categorical(tensor([[0.5, 0.25, 0.25], [0.25, 0.25, 0.5]]))


@bm.random_variable
def cat8_3():
    return Categorical(
        tensor(
            [
                [0.5, 0.25, 0.25],
                [0.25, 0.25, 0.5],
                [0.5, 0.25, 0.25],
                [0.25, 0.25, 0.5],
                [0.5, 0.25, 0.25],
                [0.25, 0.25, 0.5],
                [0.5, 0.25, 0.25],
                [0.25, 0.25, 0.5],
                [0.5, 0.25, 0.25],
                [0.25, 0.25, 0.5],
            ]
        )
    )


class NodeSupportTest(unittest.TestCase):
    def assertEqual(self, x: Any, y: Any) -> None:
        if isinstance(x, Tensor) and isinstance(y, Tensor):
            self.assertTrue(tensor_equality(x, y))
        else:
            super().assertEqual(x, y)

    def test_node_supports(self) -> None:
        self.maxDiff = None

        rt = BMGRuntime()
        rt.accumulate_graph([and1(), negexp1()], {})
        cs = ComputeSupport()

        expected_flip1 = """
tensor(0.)
tensor(1.)"""
        observed_flip1 = str(cs[rt._rv_to_node(flip1(0))])
        self.assertEqual(expected_flip1.strip(), observed_flip1.strip())

        expected_sum1 = """
tensor(1.)
tensor(2.)"""
        observed_sum1 = str(cs[rt._rv_to_node(sum1())])
        self.assertEqual(expected_sum1.strip(), observed_sum1.strip())

        expected_prod1 = """
tensor(1.)
tensor(2.)
tensor(4.)"""
        observed_prod1 = str(cs[rt._rv_to_node(prod1())])
        self.assertEqual(expected_prod1.strip(), observed_prod1.strip())

        expected_pow1 = """
tensor(1.)
tensor(16.)
tensor(2.)
tensor(256.)
tensor(4.)
"""
        observed_pow1 = str(cs[rt._rv_to_node(pow1())])
        self.assertEqual(expected_pow1.strip(), observed_pow1.strip())

        expected_ge1 = """
tensor(False)
tensor(True)
"""
        observed_ge1 = str(cs[rt._rv_to_node(ge1())])
        self.assertEqual(expected_ge1.strip(), observed_ge1.strip())

        expected_and1 = expected_ge1
        observed_and1 = str(cs[rt._rv_to_node(and1())])
        self.assertEqual(expected_and1.strip(), observed_and1.strip())

        # Some versions of torch display -exp(4) as -54.5981, and some display it
        # as -54.5982. (The actual value is -54.5981500331..., which is not an excuse
        # for some versions getting it wrong.)  To avoid this test randomly failing
        # depending on which version of torch we're using, we'll truncate to integers.

        expected_exp1 = "['-2', '-54', '-7']"
        results = [str(int(t)) for t in cs[rt._rv_to_node(negexp1())]]
        results.sort()
        self.assertEqual(expected_exp1.strip(), str(results).strip())

    def test_bernoulli_support(self) -> None:

        self.maxDiff = None

        rt = BMGRuntime()
        rt.accumulate_graph([flip2(0)], {})
        sample = rt._rv_to_node(flip2(0))
        s = ComputeSupport()
        observed = str(s[sample])
        expected = """
tensor([[0., 0.]])
tensor([[0., 1.]])
tensor([[1., 0.]])
tensor([[1., 1.]])"""

        self.assertEqual(expected.strip(), observed.strip())

    def test_categorical_support(self) -> None:

        self.maxDiff = None

        rt = BMGRuntime()
        rt.accumulate_graph([cat3(), cat2_3(), cat8_3()], {})
        s = ComputeSupport()

        c3 = rt._rv_to_node(cat3())
        observed_c3 = str(s[c3])
        expected_c3 = """
tensor(0)
tensor(1)
tensor(2)
"""
        self.assertEqual(expected_c3.strip(), observed_c3.strip())

        c23 = rt._rv_to_node(cat2_3())
        observed_c23 = str(s[c23])
        expected_c23 = """
tensor([0, 0])
tensor([0, 1])
tensor([0, 2])
tensor([1, 0])
tensor([1, 1])
tensor([1, 2])
tensor([2, 0])
tensor([2, 1])
tensor([2, 2])
"""
        self.assertEqual(expected_c23.strip(), observed_c23.strip())

        c83 = rt._rv_to_node(cat8_3())
        observed_c23 = s[c83]
        self.assertTrue(observed_c23 is TooBig)

    def test_stochastic_tensor_support(self) -> None:
        self.maxDiff = None

        rt = BMGRuntime()
        rt.accumulate_graph([to_tensor()], {})
        tm = rt._rv_to_node(to_tensor())
        s = ComputeSupport()
        observed = str(s[tm])
        expected = """
tensor([2.5000, 0.0000, 0.0000, 0.0000])
tensor([2.5000, 0.0000, 0.0000, 1.0000])
tensor([2.5000, 0.0000, 1.0000, 0.0000])
tensor([2.5000, 0.0000, 1.0000, 1.0000])
tensor([2.5000, 1.0000, 0.0000, 0.0000])
tensor([2.5000, 1.0000, 0.0000, 1.0000])
tensor([2.5000, 1.0000, 1.0000, 0.0000])
tensor([2.5000, 1.0000, 1.0000, 1.0000])
"""

        self.assertEqual(expected.strip(), observed.strip())

    def test_infinite_support(self) -> None:
        self.maxDiff = None
        rt = BMGRuntime()
        rt.accumulate_graph([normal()], {})
        sample = rt._rv_to_node(normal())
        s = ComputeSupport()
        observed = s[sample]
        self.assertEqual(Infinite, observed)
