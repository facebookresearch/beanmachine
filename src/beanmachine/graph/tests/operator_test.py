# Copyright (c) Facebook, Inc. and its affiliates.
import math
import unittest

import beanmachine.graph as bmg
import numpy as np


def tidy(s: str) -> str:
    return "\n".join(c.strip() for c in s.strip().split("\n")).strip()


class TestOperators(unittest.TestCase):
    def test_oper_args(self) -> None:
        """
        We will test test number of arguments for each operator 0, 1, 2, 3 etc.
        """

        self.maxDiff = None

        g = bmg.Graph()
        c1 = g.add_constant_real(2.5)
        c2 = g.add_constant_real(-1.5)
        c3 = g.add_constant_probability(0.5)
        c4 = g.add_constant_probability(0.6)
        c5 = g.add_constant_probability(0.7)
        c6 = g.add_constant_natural(23)
        c7 = g.add_constant_bool(False)
        c8 = g.add_constant_neg_real(-1.25)
        c9 = g.add_constant_pos_real(1.25)
        # add const matrices, operators on matrix to be added
        g.add_constant_bool_matrix(np.array([[True, False], [False, True]]))
        g.add_constant_real_matrix(np.array([[-0.1, 0.0], [2.0, -1.0]]))
        g.add_constant_natural_matrix(np.array([[1, 2], [0, 999]]))
        g.add_constant_pos_matrix(np.array([[0.1, 0.0], [2.0, 1.0]]))
        g.add_constant_neg_matrix(np.array(([-0.3, -0.4])))
        g.add_constant_probability_matrix(np.array([0.1, 0.9]))
        g.add_constant_col_simplex_matrix(np.array([[0.1, 1.0], [0.9, 0.0]]))
        with self.assertRaises(ValueError):
            g.add_constant_neg_matrix(np.array([[0.1, 0.0], [2.0, -1.0]]))
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

        observed = g.to_string()
        expected = """
Node 0 type 1 parents [ ] children [ 24 30 31 31 33 ] real 2.5
Node 1 type 1 parents [ ] children [ 19 23 30 31 33 ] real -1.5
Node 2 type 1 parents [ ] children [ 22 29 ] probability 0.5
Node 3 type 1 parents [ ] children [ 16 26 28 29 ] probability 0.6
Node 4 type 1 parents [ ] children [ 28 29 ] probability 0.7
Node 5 type 1 parents [ ] children [ 17 ] natural 23
Node 6 type 1 parents [ ] children [ 27 ] boolean 0
Node 7 type 1 parents [ ] children [ 18 20 25 32 32 32 ] negative real -1.25
Node 8 type 1 parents [ ] children [ 21 ] positive real 1.25
Node 9 type 1 parents [ ] children [ ] matrix<boolean> 1 0
0 1
Node 10 type 1 parents [ ] children [ ] matrix<real> -0.1    0
2   -1
Node 11 type 1 parents [ ] children [ ] matrix<natural>   1   2
0 999
Node 12 type 1 parents [ ] children [ ] matrix<positive real> 0.1   0
2   1
Node 13 type 1 parents [ ] children [ ] matrix<negative real> -0.3
-0.4
Node 14 type 1 parents [ ] children [ ] matrix<probability> 0.1
0.9
Node 15 type 1 parents [ ] children [ ] col_simplex_matrix<probability> 0.1   1
0.9   0
Node 16 type 3 parents [ 3 ] children [ ] real 0
Node 17 type 3 parents [ 5 ] children [ ] real 0
Node 18 type 3 parents [ 7 ] children [ ] real 0
Node 19 type 3 parents [ 1 ] children [ ] positive real 1e-10
Node 20 type 3 parents [ 7 ] children [ ] probability 1e-10
Node 21 type 3 parents [ 8 ] children [ ] positive real 1e-10
Node 22 type 3 parents [ 2 ] children [ ] negative real -1e-10
Node 23 type 3 parents [ 1 ] children [ ] real 0
Node 24 type 3 parents [ 0 ] children [ ] real 0
Node 25 type 3 parents [ 7 ] children [ ] positive real 1e-10
Node 26 type 3 parents [ 3 ] children [ ] probability 1e-10
Node 27 type 3 parents [ 6 ] children [ ] boolean 0
Node 28 type 3 parents [ 3 4 ] children [ ] probability 1e-10
Node 29 type 3 parents [ 2 3 4 ] children [ ] probability 1e-10
Node 30 type 3 parents [ 0 1 ] children [ ] real 0
Node 31 type 3 parents [ 0 1 0 ] children [ ] real 0
Node 32 type 3 parents [ 7 7 7 ] children [ ] negative real -1e-10
Node 33 type 3 parents [ 0 1 ] children [ ] real 0
        """
        self.assertEqual(tidy(expected), tidy(observed))

    def test_arithmetic(self) -> None:
        g = bmg.Graph()
        c1 = g.add_constant_natural(3)
        o0 = g.add_operator(bmg.OperatorType.TO_REAL, [c1])
        o1 = g.add_operator(bmg.OperatorType.NEGATE, [o0])
        o2 = g.add_operator(bmg.OperatorType.EXP, [o1])  # positive real
        o2_real = g.add_operator(bmg.OperatorType.TO_REAL, [o2])
        o3 = g.add_operator(bmg.OperatorType.MULTIPLY, [o2_real, o0])
        o4 = g.add_operator(bmg.OperatorType.EXPM1, [o0])
        o5 = g.add_operator(bmg.OperatorType.ADD, [o0, o3, o4])
        o6 = g.add_operator(bmg.OperatorType.POW, [o5, o0])  # real
        # Verify that EXPM1 on a negative real is legal.
        o7 = g.add_operator(bmg.OperatorType.NEGATE, [o2])
        o8 = g.add_operator(bmg.OperatorType.EXPM1, [o7])
        g.query(o6)
        g.query(o8)
        samples = g.infer(2)
        # both samples should have exactly the same value since we are doing
        # deterministic operators only
        self.assertEqual(type(samples[0][0]), float)
        self.assertEqual(samples[0][0], samples[1][0])
        # the result should be identical to doing this math directly
        const1 = 3.0
        r6 = (const1 + math.exp(-const1) * const1 + math.expm1(const1)) ** const1
        self.assertAlmostEqual(samples[0][0], r6, 3)
        r8 = math.expm1(-math.exp(-const1))
        self.assertAlmostEqual(samples[0][1], r8, 3)

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
        c0 = g.add_constant_real(0.25)
        c1 = g.add_constant_real(0.5)
        c2 = g.add_constant_real(0.75)
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

    def test_to_neg_real(self) -> None:
        # We have some situations where we know that a real quantity
        # is negative but we cannot prove it. For example,
        # log(0.4 * beta() + 0.5) is definitely negative but we
        # assume that the sum of two probabilities is a positive real,
        # and so the log is a real, not a negative real.
        #
        # The to_neg_real operator takes a real and constrains it to
        # be negative.
        g = bmg.Graph()
        two = g.add_constant_pos_real(2.0)
        beta = g.add_distribution(
            bmg.DistributionType.BETA, bmg.AtomicType.PROBABILITY, [two, two]
        )
        s = g.add_operator(bmg.OperatorType.SAMPLE, [beta])
        c4 = g.add_constant_probability(0.4)
        c5 = g.add_constant_pos_real(0.5)
        mult = g.add_operator(bmg.OperatorType.MULTIPLY, [c4, s])
        tr = g.add_operator(bmg.OperatorType.TO_POS_REAL, [mult])
        add = g.add_operator(bmg.OperatorType.ADD, [tr, c5])  # Positive real
        lg = g.add_operator(bmg.OperatorType.LOG, [add])  # Real
        tnr = g.add_operator(bmg.OperatorType.TO_NEG_REAL, [lg])
        lme = g.add_operator(bmg.OperatorType.LOG1MEXP, [tnr])
        ex = g.add_operator(bmg.OperatorType.EXP, [lme])
        g.query(add)
        g.query(lg)
        g.query(tnr)
        g.query(ex)
        samples = g.infer(1, bmg.InferenceType.NMC)[0]
        add_sample = samples[0]
        lg_sample = samples[1]
        tnr_sample = samples[2]
        ex_sample = samples[3]

        self.assertTrue(0.5 <= add_sample <= 0.9)
        self.assertTrue(lg_sample <= 0.0)
        self.assertEqual(lg_sample, tnr_sample)
        self.assertAlmostEqual(ex_sample, 1.0 - add_sample, 3)

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
