# Copyright (c) Facebook, Inc. and its affiliates.
import unittest
from typing import Any

import beanmachine.ppl as bm
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_nodes import positive_infinity
from beanmachine.ppl.compiler.runtime import BMGRuntime
from beanmachine.ppl.compiler.support import ComputeSupport, Infinite
from beanmachine.ppl.utils.set_of_tensors import SetOfTensors
from torch import Size, Tensor, tensor
from torch.distributions import Bernoulli, Normal

# TODO: The first half of this file tests the support instance method
# on nodes which we are going to delete and replace with a nonrecursive
# support computation in its own module.  Delete these tests once
# that is ready.


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


class NodeSupportTest(unittest.TestCase):
    def assertEqual(self, x: Any, y: Any) -> bool:
        if isinstance(x, Tensor) and isinstance(y, Tensor):
            return tensor_equality(x, y)
        return super().assertEqual(x, y)

    def test_node_supports(self) -> None:
        # TODO: More tests.
        bmg = BMGraphBuilder()
        t5 = tensor(0.5)
        t1 = tensor(1.0)
        t2 = tensor(2.0)
        t0 = tensor(0.0)
        t = bmg.add_constant_tensor(t5)
        bern = bmg.add_bernoulli(t)
        s = bmg.add_sample(bern)
        a1 = bmg.add_addition(s, t)
        a2 = bmg.add_addition(s, s)
        self.assertEqual(SetOfTensors(t.support()), SetOfTensors([t5]))
        self.assertEqual(SetOfTensors(s.support()), SetOfTensors([t0, t1]))
        self.assertEqual(SetOfTensors(a1.support()), SetOfTensors([t0 + t5, t1 + t5]))
        self.assertEqual(SetOfTensors(a2.support()), SetOfTensors([t0, t1, t2]))

    def test_node_support_sizes(self) -> None:
        bmg = BMGraphBuilder()
        c = bmg.add_constant(2.5)
        self.assertEqual(c.support_size(), 1)
        bern = bmg.add_bernoulli(bmg.add_constant_tensor(tensor(0.5)))
        self.assertEqual(bern.support_size(), 2)
        berns = bmg.add_sample(bern)
        self.assertEqual(berns.support_size(), 2)
        self.assertTrue(berns.support_size() >= len(list(berns.support())))
        # bern() + 2.5 has two possible values
        a = bmg.add_addition(berns, c)
        self.assertEqual(a.support_size(), 2)
        self.assertTrue(a.support_size() >= len(list(a.support())))
        # bern() * bern() has two possible values but we think it is four
        # because we do not know they are constrained to be equal.
        m = bmg.add_multiplication(berns, berns)
        self.assertEqual(m.support_size(), 4)
        self.assertTrue(m.support_size() >= len(list(m.support())))
        # Similarly [bern(), bern()] has two possible values but we think four.
        t = bmg.add_tensor(Size([2]), berns, berns)
        self.assertEqual(t.support_size(), 4)
        self.assertTrue(t.support_size() >= len(list(t.support())))
        cat = bmg.add_categorical(bmg.add_constant_tensor(tensor([0.5, 0.25, 0.25])))
        self.assertEqual(cat.support_size(), 3)
        self.assertTrue(cat.support_size() >= len(list(cat.support())))
        n = bmg.add_normal(c, c)
        self.assertEqual(n.support_size(), positive_infinity)
        n1 = bmg.add_sample(n)
        self.assertEqual(n1.support_size(), positive_infinity)
        n2 = bmg.add_sample(n)
        self.assertEqual(n2.support_size(), positive_infinity)
        # The support of a comparison is two, even if the operands have
        # infinite support
        gt = bmg.add_greater_than(n1, n2)
        self.assertEqual(gt.support_size(), 2)

        # TODO: There's a bug in gt.support(); it should return { True, False }
        # TODO: Disable the next line of the test until we fix that.
        # TODO: self.assertTrue(gt.support_size() >= len(list(gt.support())))

        # Exp(bool) has support two.
        eb = bmg.add_exp(berns)
        self.assertEqual(eb.support_size(), 2)
        self.assertTrue(eb.support_size() >= len(list(eb.support())))
        # Exp(normal) has infinite support
        en = bmg.add_exp(n1)
        self.assertEqual(en.support_size(), positive_infinity)

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
