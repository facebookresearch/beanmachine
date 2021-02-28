# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch.distributions import Bernoulli


# TODO: x > y   -->  if x then not y else false
# TODO: x is y  -->  same as == ? Or should this be illegal?


@bm.random_variable
def flip(n):
    return Bernoulli(0.5)


#
# ==
#


@bm.functional
def eq_x_0():
    # not flip(0)
    return flip(0) == 0.0


@bm.functional
def eq_x_1():
    # flip(0)
    return flip(0) == 1.0


@bm.functional
def eq_0_y():
    # not flip(1)
    return 0 == flip(1)


@bm.functional
def eq_1_y():
    # flip(1)
    return 1 == flip(1)


@bm.functional
def eq_x_y():
    # if flip(0) then flip(1) else not flip(1)
    return flip(0) == flip(1)


#
# !=
#


@bm.functional
def neq_x_0():
    # flip(0)
    return flip(0) != 0.0


@bm.functional
def neq_x_1():
    # not flip(0)
    return flip(0) != 1.0


@bm.functional
def neq_0_y():
    # flip(1)
    return 0 != flip(1)


@bm.functional
def neq_1_y():
    # not flip(1)
    return 1 != flip(1)


@bm.functional
def neq_x_y():
    # if flip(0) then not flip(1) else flip(1)
    return flip(0) != flip(1)


#
# >=
#


@bm.functional
def gte_x_0():
    # true
    return flip(0) >= 0.0


@bm.functional
def gte_x_1():
    # flip(0)
    return flip(0) >= 1.0


@bm.functional
def gte_0_y():
    # not flip(1)
    return 0 >= flip(1)


@bm.functional
def gte_1_y():
    # true
    return 1 >= flip(1)


@bm.functional
def gte_x_y():
    # if flip(0) then true else not flip(1)
    return flip(0) >= flip(1)


#
# <=
#


@bm.functional
def lte_x_0():
    # not flip(0)
    return flip(0) <= 0.0


@bm.functional
def lte_x_1():
    # true
    return flip(0) <= 1.0


@bm.functional
def lte_0_y():
    # true
    return 0 <= flip(1)


@bm.functional
def lte_1_y():
    # flip(1)
    return 1 <= flip(1)


@bm.functional
def lte_x_y():
    # if flip(0) then flip(1) else true
    return flip(0) <= flip(1)


#
# <
#


@bm.functional
def lt_x_0():
    # false
    return flip(0) < 0.0


@bm.functional
def lt_x_1():
    # not flip(0)
    return flip(0) < 1.0


@bm.functional
def lt_0_y():
    # flip(1)
    return 0 < flip(1)


@bm.functional
def lt_1_y():
    # false
    return 1 < flip(1)


@bm.functional
def lt_x_y():
    # if flip(0) then false else flip(1)
    return flip(0) < flip(1)


#
# >
#


@bm.functional
def gt_x_0():
    # flip(0)
    return flip(0) > 0.0


@bm.functional
def gt_x_1():
    # false
    return flip(0) > 1.0


@bm.functional
def gt_0_y():
    # false
    return 0 > flip(1)


@bm.functional
def gt_1_y():
    # not flip(1)
    return 1 > flip(1)


@bm.functional
def gt_x_y():
    # if flip(0) then not flip(1) else false
    return flip(0) > flip(1)


class BooleanComparisonsTest(unittest.TestCase):
    def test_boolean_comparison_eq(self) -> None:

        self.maxDiff = None

        observed = BMGInference().to_dot([eq_x_y()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Sample];
  N4[label=complement];
  N5[label=if];
  N6[label=Query];
  N0 -> N1;
  N1 -> N2;
  N1 -> N3;
  N2 -> N5;
  N3 -> N4;
  N3 -> N5;
  N4 -> N5;
  N5 -> N6;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([eq_x_0()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=complement];
  N4[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
        observed = BMGInference().to_dot([eq_0_y()], {})
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([eq_x_1()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
        observed = BMGInference().to_dot([eq_1_y()], {})
        self.assertEqual(expected.strip(), observed.strip())

    def test_boolean_comparison_neq(self) -> None:

        self.maxDiff = None

        observed = BMGInference().to_dot([neq_x_y()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Sample];
  N4[label=complement];
  N5[label=if];
  N6[label=Query];
  N0 -> N1;
  N1 -> N2;
  N1 -> N3;
  N2 -> N5;
  N3 -> N4;
  N3 -> N5;
  N4 -> N5;
  N5 -> N6;
}
        """
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([neq_x_0()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
        observed = BMGInference().to_dot([neq_0_y()], {})
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([neq_x_1()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=complement];
  N4[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
        observed = BMGInference().to_dot([neq_1_y()], {})
        self.assertEqual(expected.strip(), observed.strip())

    def test_boolean_comparison_gte(self) -> None:

        self.maxDiff = None

        observed = BMGInference().to_dot([gte_x_y()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Sample];
  N4[label=True];
  N5[label=complement];
  N6[label=if];
  N7[label=Query];
  N0 -> N1;
  N1 -> N2;
  N1 -> N3;
  N2 -> N6;
  N3 -> N5;
  N4 -> N6;
  N5 -> N6;
  N6 -> N7;
}
        """
        self.assertEqual(expected.strip(), observed.strip())

        # TODO: Note that here we keep the sample in the graph even though it is
        # not queried or observed. We might consider removing it.
        observed = BMGInference().to_dot([gte_x_0()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=True];
  N4[label=Query];
  N0 -> N1;
  N1 -> N2;
  N3 -> N4;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([gte_0_y()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=complement];
  N4[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
}
"""

        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([gte_x_1()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([gte_1_y()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=True];
  N4[label=Query];
  N0 -> N1;
  N1 -> N2;
  N3 -> N4;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_boolean_comparison_lte(self) -> None:

        self.maxDiff = None

        observed = BMGInference().to_dot([lte_x_y()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Sample];
  N4[label=True];
  N5[label=if];
  N6[label=Query];
  N0 -> N1;
  N1 -> N2;
  N1 -> N3;
  N2 -> N5;
  N3 -> N5;
  N4 -> N5;
  N5 -> N6;
}
        """
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([lte_x_0()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=complement];
  N4[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([lte_0_y()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=True];
  N4[label=Query];
  N0 -> N1;
  N1 -> N2;
  N3 -> N4;
}
"""

        self.assertEqual(expected.strip(), observed.strip())
        observed = BMGInference().to_dot([lte_x_1()], {})
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([lte_1_y()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_boolean_comparison_lt(self) -> None:

        self.maxDiff = None

        observed = BMGInference().to_dot([lt_x_y()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Sample];
  N4[label=False];
  N5[label=if];
  N6[label=Query];
  N0 -> N1;
  N1 -> N2;
  N1 -> N3;
  N2 -> N5;
  N3 -> N5;
  N4 -> N5;
  N5 -> N6;
}"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([lt_x_0()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=False];
  N4[label=Query];
  N0 -> N1;
  N1 -> N2;
  N3 -> N4;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
        observed = BMGInference().to_dot([lt_1_y()], {})
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([lt_0_y()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
}
"""

        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([lt_x_1()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=complement];
  N4[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
}
"""

        self.assertEqual(expected.strip(), observed.strip())

    def test_boolean_comparison_gt(self) -> None:

        self.maxDiff = None

        observed = BMGInference().to_dot([gt_x_y()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Sample];
  N4[label=complement];
  N5[label=False];
  N6[label=if];
  N7[label=Query];
  N0 -> N1;
  N1 -> N2;
  N1 -> N3;
  N2 -> N6;
  N3 -> N4;
  N4 -> N6;
  N5 -> N6;
  N6 -> N7;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([gt_x_0()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([gt_1_y()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=complement];
  N4[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([gt_0_y()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=False];
  N4[label=Query];
  N0 -> N1;
  N1 -> N2;
  N3 -> N4;
}
"""

        self.assertEqual(expected.strip(), observed.strip())
        observed = BMGInference().to_dot([gt_x_1()], {})
        self.assertEqual(expected.strip(), observed.strip())
