# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.graph as bmg
import torch


class TestOperators(unittest.TestCase):
    def test_oper_args(self):
        """
        We will test test number of arguments for each operator 0, 1, 2, 3 etc.
        """
        g = bmg.Graph()
        c1 = g.add_constant(torch.FloatTensor([0, 1, -1]))
        c2 = g.add_constant(torch.FloatTensor([0, 1, -1]))
        c3 = g.add_constant(torch.FloatTensor([0, 1, -1]))
        # test TO_REAL
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.TO_REAL, [])
        g.add_operator(bmg.OperatorType.TO_REAL, [c1])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.TO_REAL, [c1, c2])
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

    def test_arithmetic(self):
        g = bmg.Graph()
        const1 = torch.FloatTensor([0, 1, -1])
        c1 = g.add_constant(const1)
        o1 = g.add_operator(bmg.OperatorType.NEGATE, [c1])
        o2 = g.add_operator(bmg.OperatorType.EXP, [o1])
        o3 = g.add_operator(bmg.OperatorType.MULTIPLY, [o2, c1])
        o4 = g.add_operator(bmg.OperatorType.ADD, [c1, c1, o3])
        g.query(o4)
        samples = g.infer(2)
        # both samples should have exactly the same value since we are doing
        # deterministic operators only
        self.assertEqual(samples[0][0].type, bmg.AtomicType.TENSOR)
        self.assertTrue((samples[0][0].tensor == samples[1][0].tensor).all().item())
        # the result should be identical to doing this math directly on tensors
        result = const1 + const1 + torch.exp(-const1) * const1
        self.assertTrue((samples[0][0].tensor == result).all().item())

    def test_sample(self):
        # negative test we can't exponentiate the sample from a Bernoulli
        g = bmg.Graph()
        c1 = g.add_constant(0.6)
        d1 = g.add_distribution(
            bmg.DistributionType.BERNOULLI, bmg.AtomicType.BOOLEAN, [c1]
        )
        s1 = g.add_operator(bmg.OperatorType.SAMPLE, [d1])
        o1 = g.add_operator(bmg.OperatorType.EXP, [s1])
        g.query(o1)
        # note: this error is raised by the torch library so we don't want to
        # assert on the error message here
        with self.assertRaises(RuntimeError) as cm:
            g.infer(1)
        self.assertTrue("invalid parent type" in str(cm.exception))

        # the proper way to do it is to convert to floating point first
        g = bmg.Graph()
        c1 = g.add_constant(0.6)
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
        self.assertEqual(samples[0][0].type, bmg.AtomicType.REAL)
        self.assertEqual(
            [s[0].real for s in samples], [0.0] * 10, "all samples should be zero"
        )
