# Copyright (c) Facebook, Inc. and its affiliates.
import math
import unittest

import beanmachine.graph as bmg
import numpy as np


class TestOperators(unittest.TestCase):
    def test_oper_args(self) -> None:
        """
        We will test test number of arguments for each operator 0, 1, 2, 3 etc.
        """
        g = bmg.Graph()
        c1 = g.add_constant(2.5)
        c2 = g.add_constant(-1.5)
        c3 = g.add_constant_probability(0.5)
        c4 = g.add_constant_probability(0.6)
        c5 = g.add_constant_probability(0.7)
        c6 = g.add_constant(23)  # NATURAL
        c7 = g.add_constant(False)
        c8 = g.add_constant_neg_real(-1.25)
        c9 = g.add_constant_pos_real(1.25)
        # add const matrices, operators on matrix to be added
        g.add_constant_matrix(np.array([[True, False], [False, True]]))
        g.add_constant_matrix(np.array([[-0.1, 0.0], [2.0, -1.0]]))
        g.add_constant_matrix(np.array([[1, 2], [0, 999]]))
        g.add_constant_pos_matrix(np.array([[0.1, 0.0], [2.0, 1.0]]))
        g.add_constant_probability_matrix(np.array([0.1, 0.9]))
        g.add_constant_col_simplex_matrix(np.array([[0.1, 1.0], [0.9, 0.0]]))
        with self.assertRaises(ValueError):
            g.add_constant_pos_matrix(np.array([[0.1, 0.0], [2.0, -1.0]]))
        with self.assertRaises(ValueError):
            g.add_constant_col_simplex_matrix(np.array([[0.1, 0.0], [2.0, 1.0]]))
        with self.assertRaises(ValueError):
            g.add_constant_probability_matrix(np.array([1.1, 0.9]))
        # test TO_REAL
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.TO_REAL, [])
        g.add_operator(bmg.OperatorType.TO_REAL, [c4])
        g.add_operator(bmg.OperatorType.TO_REAL, [c6])
        g.add_operator(bmg.OperatorType.TO_REAL, [c8])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.TO_REAL, [c4, c5])
        # test EXP
        # Exp needs exactly one operand
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.EXP, [])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.EXP, [c2, c8])
        # That operand must be real, negative real or positive real:
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.EXP, [c3])  # prob throws
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.EXP, [c6])  # natural throws
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.EXP, [c7])  # bool throws
        g.add_operator(bmg.OperatorType.EXP, [c2])  # real OK
        g.add_operator(bmg.OperatorType.EXP, [c8])  # neg_real OK
        g.add_operator(bmg.OperatorType.EXP, [c9])  # pos_real OK
        # test LOG
        # Log needs exactly one operand:
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.LOG, [])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.LOG, [c1, c2])
        # That operand must be positive real or probability:
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.LOG, [c2])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.LOG, [c6])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.LOG, [c7])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.LOG, [c8])
        g.add_operator(bmg.OperatorType.LOG, [c3])
        # test NEGATE
        # Negate needs exactly one operand
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.NEGATE, [])
        g.add_operator(bmg.OperatorType.NEGATE, [c2])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.NEGATE, [c1, c2])
        # Negate can take a real, negative real or positive real.
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.NEGATE, [c3])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.NEGATE, [c6])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.NEGATE, [c7])
        g.add_operator(bmg.OperatorType.NEGATE, [c1])
        g.add_operator(bmg.OperatorType.NEGATE, [c8])
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
            g.add_operator(bmg.OperatorType.MULTIPLY, [c3])
        g.add_operator(bmg.OperatorType.MULTIPLY, [c4, c5])
        g.add_operator(bmg.OperatorType.MULTIPLY, [c3, c4, c5])
        # test ADD
        # Add requires two or more operands
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.ADD, [])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.ADD, [c1])
        # All operands must be (1) the same type, and (2)
        # real, neg real or pos real.
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.ADD, [c1, c8])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.ADD, [c3, c3])
        g.add_operator(bmg.OperatorType.ADD, [c1, c2])
        g.add_operator(bmg.OperatorType.ADD, [c1, c2, c1])
        g.add_operator(bmg.OperatorType.ADD, [c8, c8, c8])
        # test POW
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.POW, [])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.POW, [c1])
        with self.assertRaises(ValueError):
            g.add_operator(bmg.OperatorType.POW, [c1, c1, c1])
        g.add_operator(bmg.OperatorType.POW, [c1, c2])

    def test_arithmetic(self) -> None:
        g = bmg.Graph()
        c1 = g.add_constant(3)  # natural
        o0 = g.add_operator(bmg.OperatorType.TO_REAL, [c1])
        o1 = g.add_operator(bmg.OperatorType.NEGATE, [o0])
        o2 = g.add_operator(bmg.OperatorType.EXP, [o1])
        o2_real = g.add_operator(bmg.OperatorType.TO_REAL, [o2])
        o3 = g.add_operator(bmg.OperatorType.MULTIPLY, [o2_real, o0])
        o4 = g.add_operator(bmg.OperatorType.EXPM1, [o0])
        o5 = g.add_operator(bmg.OperatorType.ADD, [o0, o3, o4])
        o6 = g.add_operator(bmg.OperatorType.POW, [o5, o0])  # real
        g.query(o6)
        samples = g.infer(2)
        # both samples should have exactly the same value since we are doing
        # deterministic operators only
        self.assertEqual(type(samples[0][0]), float)
        self.assertEqual(samples[0][0], samples[1][0])
        # the result should be identical to doing this math directly
        const1 = 3.0
        result = (const1 + math.exp(-const1) * const1 + math.expm1(const1)) ** const1
        self.assertAlmostEqual(samples[0][0], result, 3)

    def test_probability(self) -> None:
        g = bmg.Graph()
        c1 = g.add_constant_probability(0.8)
        c2 = g.add_constant_probability(0.7)
        o1 = g.add_operator(bmg.OperatorType.COMPLEMENT, [c1])
        o2 = g.add_operator(bmg.OperatorType.MULTIPLY, [o1, c2])
        g.query(o2)
        samples = g.infer(2)
        self.assertTrue(type(samples[0][0]), float)
        self.assertAlmostEqual(samples[0][0], 0.14, 3)

    def test_to_probability(self) -> None:
        # We have some situations where we know that a real or positive
        # real quantity is a probability but we cannot prove it. For
        # example, 0.4 * beta_sample + 0.5 is definitely between 0.0 and
        # 1.0, but we assume that the sum of two probabilities is a
        # positive real.
        #
        # The to_probability operator takes a real or positive real and
        # constrains it to the range (0.0, 1.0)

        g = bmg.Graph()
        c0 = g.add_constant(0.25)
        c1 = g.add_constant(0.5)
        c2 = g.add_constant(0.75)
        o0 = g.add_operator(bmg.OperatorType.ADD, [c0, c1])
        o1 = g.add_operator(bmg.OperatorType.TO_PROBABILITY, [o0])
        o2 = g.add_operator(bmg.OperatorType.ADD, [c1, c2])
        o3 = g.add_operator(bmg.OperatorType.TO_PROBABILITY, [o2])
        g.query(o0)
        g.query(o1)
        g.query(o2)
        g.query(o3)
        samples = g.infer(1)
        self.assertAlmostEqual(samples[0][0], 0.75, 3)
        self.assertAlmostEqual(samples[0][1], 0.75, 3)
        self.assertAlmostEqual(samples[0][2], 1.25, 3)
        self.assertAlmostEqual(samples[0][3], 1.0, 3)

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
            "operator EXP requires a neg_real, real or pos_real parent"
            in str(cm.exception)
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
        # direcly negating o3 results in a NEG_REAL value
        g.add_operator(bmg.OperatorType.NEGATE, [o3])
        # converting o3 to REAL then applying negate results in REAL value
        o3_real = g.add_operator(bmg.OperatorType.TO_REAL, [o3])
        o4 = g.add_operator(bmg.OperatorType.NEGATE, [o3_real])
        o2_real = g.add_operator(bmg.OperatorType.TO_REAL, [o2])
        o5 = g.add_operator(bmg.OperatorType.ADD, [o2_real, o4])
        # o5 should be 0 in all possible worlds
        g.query(o5)
        samples = g.infer(10)
        self.assertEqual(type(samples[0][0]), float)
        self.assertEqual(
            [s[0] for s in samples], [0.0] * 10, "all samples should be zero"
        )
