# Copyright (c) Facebook, Inc. and its affiliates.
import torch  # isort:skip  torch has to be imported before graph
import math
import unittest

import beanmachine.graph as bmg


class TestOperators(unittest.TestCase):
    def test_oper_args(self) -> None:
        """
        We will test test number of arguments for each operator 0, 1, 2, 3 etc.
        """
        g = bmg.Graph()
        c1 = g.add_constant(torch.FloatTensor([0, 1, -1]))
        c2 = g.add_constant(torch.FloatTensor([0, 1, -1]))
        c3 = g.add_constant(torch.FloatTensor([0, 1, -1]))
        c4 = g.add_constant_probability(0.6)
        c5 = g.add_constant_probability(0.7)
        c6 = g.add_constant(23)  # NATURAL
        c7 = g.add_constant(False)
        # test TO_REAL
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.TO_REAL, [])
        with self.assertRaises(ValueError):
            # can't convert tensor to real
            g.add_operator(bmg.OperatorType.TO_REAL, [c1])
        g.add_operator(bmg.OperatorType.TO_REAL, [c4])
        g.add_operator(bmg.OperatorType.TO_REAL, [c6])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.TO_REAL, [c4, c5])
        # test TO_TENSOR
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.TO_TENSOR, [])
        g.add_operator(bmg.OperatorType.TO_TENSOR, [c1])
        g.add_operator(bmg.OperatorType.TO_TENSOR, [c4])
        g.add_operator(bmg.OperatorType.TO_TENSOR, [c6])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.TO_TENSOR, [c1, c2])
        # test EXP
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.EXP, [])
        g.add_operator(bmg.OperatorType.EXP, [c1])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.EXP, [c1, c2])
        # test NEGATE
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.NEGATE, [])
        g.add_operator(bmg.OperatorType.NEGATE, [c1])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.NEGATE, [c1, c2])
        # test COMPLEMENT
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.COMPLEMENT, [])
        g.add_operator(bmg.OperatorType.COMPLEMENT, [c4])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.COMPLEMENT, [c4, c4])
        g.add_operator(bmg.OperatorType.COMPLEMENT, [c7])
        # test MULTIPLY
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.MULTIPLY, [])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.MULTIPLY, [c1])
        g.add_operator(bmg.OperatorType.MULTIPLY, [c1, c2])
        g.add_operator(bmg.OperatorType.MULTIPLY, [c1, c2, c3])
        # test ADD
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.ADD, [])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.ADD, [c1])
        g.add_operator(bmg.OperatorType.ADD, [c1, c2])
        g.add_operator(bmg.OperatorType.ADD, [c1, c2, c3])

    def test_arithmetic(self) -> None:
        g = bmg.Graph()
        c1 = g.add_constant(3)  # natural
        o0 = g.add_operator(bmg.OperatorType.TO_REAL, [c1])
        o1 = g.add_operator(bmg.OperatorType.NEGATE, [o0])
        o2 = g.add_operator(bmg.OperatorType.EXP, [o1])
        o3 = g.add_operator(bmg.OperatorType.MULTIPLY, [o2, o0])
        o4 = g.add_operator(bmg.OperatorType.EXPM1, [o0])
        o5 = g.add_operator(bmg.OperatorType.ADD, [o0, o3, o4])
        g.query(o5)
        samples = g.infer(2)
        # both samples should have exactly the same value since we are doing
        # deterministic operators only
        self.assertEqual(type(samples[0][0]), float)
        self.assertEqual(samples[0][0], samples[1][0])
        # the result should be identical to doing this math directly on tensors
        const1 = 3.0
        result = const1 + math.exp(-const1) * const1 + math.expm1(const1)
        self.assertAlmostEqual(samples[0][0], result, 3)

    def test_tensor_arithmetic(self) -> None:
        g = bmg.Graph()
        const1 = torch.FloatTensor([0, 1, -1])
        c1 = g.add_constant(const1)
        o0 = g.add_operator(bmg.OperatorType.TO_TENSOR, [c1])
        o1 = g.add_operator(bmg.OperatorType.NEGATE, [o0])
        o2 = g.add_operator(bmg.OperatorType.EXP, [o1])
        o3 = g.add_operator(bmg.OperatorType.MULTIPLY, [o2, c1])
        o4 = g.add_operator(bmg.OperatorType.EXPM1, [c1])
        o5 = g.add_operator(bmg.OperatorType.ADD, [c1, o3, o4])
        g.query(o5)
        samples = g.infer(2)
        # both samples should have exactly the same value since we are doing
        # deterministic operators only
        self.assertEqual(type(samples[0][0]), torch.Tensor)
        self.assertTrue((samples[0][0] == samples[1][0]).all().item())
        # the result should be identical to doing this math directly on tensors
        result = const1 + torch.exp(-const1) * const1 + torch.expm1(const1)
        self.assertTrue((samples[0][0] == result).all().item())

    def test_probability(self) -> None:
        g = bmg.Graph()
        c1 = g.add_constant_probability(0.8)
        c2 = g.add_constant_probability(0.7)
        o1 = g.add_operator(bmg.OperatorType.COMPLEMENT, [c1])
        o2 = g.add_operator(bmg.OperatorType.MULTIPLY, [o1, c2])
        g.query(o2)
        samples = g.infer(2)
        self.assertTrue(type(samples[0][0]), float)
        self.assertAlmostEquals(samples[0][0], 0.14, 3)

    def test_sample(self) -> None:
        # negative test we can't exponentiate the sample from a Bernoulli
        g = bmg.Graph()
        c1 = g.add_constant_probability(0.6)
        d1 = g.add_distribution(
            bmg.DistributionType.BERNOULLI, bmg.AtomicType.BOOLEAN, [c1]
        )
        s1 = g.add_operator(bmg.OperatorType.SAMPLE, [d1])
        with self.assertRaises(ValueError) as cm:
            o1 = g.add_operator(bmg.OperatorType.EXP, [s1])
        self.assertTrue(
            "operator EXP/EXPM1 requires real/tensor parent" in str(cm.exception)
        )

        # the proper way to do it is to convert to floating point first
        g = bmg.Graph()
        c1 = g.add_constant_probability(0.6)
        d1 = g.add_distribution(
            bmg.DistributionType.BERNOULLI, bmg.AtomicType.BOOLEAN, [c1]
        )
        s1 = g.add_operator(bmg.OperatorType.SAMPLE, [d1])
        o1 = g.add_operator(bmg.OperatorType.TO_REAL, [s1])
        # o2 and o3 both compute the same value
        o2 = g.add_operator(bmg.OperatorType.EXP, [o1])
        o3 = g.add_operator(bmg.OperatorType.EXP, [o1])
        o4 = g.add_operator(bmg.OperatorType.NEGATE, [o3])
        o5 = g.add_operator(bmg.OperatorType.ADD, [o2, o4])
        # o5 should be 0 in all possible worlds
        g.query(o5)
        samples = g.infer(10)
        self.assertEqual(type(samples[0][0]), float)
        self.assertEqual(
            [s[0] for s in samples], [0.0] * 10, "all samples should be zero"
        )
