# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.gen_dot import to_dot
from beanmachine.ppl.model.rv_identifier import RVIdentifier


def _rv_id() -> RVIdentifier:
    return RVIdentifier(lambda a, b: a, (1, 1))


def construct_model_graph(is_nested_cons: bool = True):
    bmg = BMGraphBuilder()
    zero = bmg.add_pos_real(0.0)
    one = bmg.add_pos_real(1.0)
    two = bmg.add_pos_real(2.0)
    three = bmg.add_pos_real(3.0)
    normal_one = bmg.add_normal(three, three)
    normal_two = bmg.add_normal(one, two)
    sample_normal_one = bmg.add_sample(normal_one)
    sample_normal_two = bmg.add_sample(normal_two)

    half = bmg.add_probability(0.5)
    bernoulli = bmg.add_bernoulli(half)
    bern_sample = bmg.add_sample(bernoulli)

    norm_if = bmg.add_if_then_else(bern_sample, sample_normal_one, sample_normal_two)

    if is_nested_cons:
        bern_if = bmg.add_if_then_else(bern_sample, norm_if, zero)
    else:
        bern_if = bmg.add_if_then_else(bern_sample, zero, norm_if)

    scale_two = bmg.add_multiplication(bern_if, two)
    bmg.add_query(scale_two, _rv_id())

    return bmg


class FixIfTest(unittest.TestCase):
    def test_nested_if_cons_fix(self) -> None:
        # This test case checks the nested if fixer for the cons case
        # IF(COND, IF(COND, CONS2, ALT2), ALT1)  --> IF(COND, CONS2, ALT1)

        self.maxDiff = None

        bmg = construct_model_graph(is_nested_cons=True)
        observed_before = to_dot(bmg, after_transform=False, label_edges=True)
        expected_before = """
digraph "graph" {
  N00[label=0.5];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=3.0];
  N04[label=Normal];
  N05[label=Sample];
  N06[label=1.0];
  N07[label=2.0];
  N08[label=Normal];
  N09[label=Sample];
  N10[label=if];
  N11[label=0.0];
  N12[label=if];
  N13[label="*"];
  N14[label=Query];
  N00 -> N01[label=probability];
  N01 -> N02[label=operand];
  N02 -> N10[label=condition];
  N02 -> N12[label=condition];
  N03 -> N04[label=mu];
  N03 -> N04[label=sigma];
  N04 -> N05[label=operand];
  N05 -> N10[label=consequence];
  N06 -> N08[label=mu];
  N07 -> N08[label=sigma];
  N07 -> N13[label=right];
  N08 -> N09[label=operand];
  N09 -> N10[label=alternative];
  N10 -> N12[label=consequence];
  N11 -> N12[label=alternative];
  N12 -> N13[label=left];
  N13 -> N14[label=operator];
}
"""
        self.assertEqual(observed_before.strip(), expected_before.strip())
        observed = to_dot(bmg, after_transform=True, label_edges=True)
        expected = """
digraph "graph" {
  N00[label=0.5];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=3.0];
  N04[label=3.0];
  N05[label=Normal];
  N06[label=Sample];
  N07[label=1.0];
  N08[label=2.0];
  N09[label=Normal];
  N10[label=Sample];
  N11[label=0.0];
  N12[label=if];
  N13[label=2.0];
  N14[label="*"];
  N15[label=Query];
  N00 -> N01[label=probability];
  N01 -> N02[label=operand];
  N02 -> N12[label=condition];
  N03 -> N05[label=mu];
  N04 -> N05[label=sigma];
  N05 -> N06[label=operand];
  N06 -> N12[label=consequence];
  N07 -> N09[label=mu];
  N08 -> N09[label=sigma];
  N09 -> N10[label=operand];
  N11 -> N12[label=alternative];
  N12 -> N14[label=left];
  N13 -> N14[label=right];
  N14 -> N15[label=operator];
}
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_nested_if_alt_fix(self) -> None:
        # This test case checks the nested if fixer for the alt case
        # IF(COND, CONS_1, IF(COND, CONS2, ALT2)) --> IF(COND, CONS1, ALT2)

        self.maxDiff = None

        bmg = construct_model_graph(is_nested_cons=False)
        observed_before = to_dot(bmg, after_transform=False, label_edges=True)
        expected_before = """
digraph "graph" {
  N00[label=0.5];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=0.0];
  N04[label=3.0];
  N05[label=Normal];
  N06[label=Sample];
  N07[label=1.0];
  N08[label=2.0];
  N09[label=Normal];
  N10[label=Sample];
  N11[label=if];
  N12[label=if];
  N13[label="*"];
  N14[label=Query];
  N00 -> N01[label=probability];
  N01 -> N02[label=operand];
  N02 -> N11[label=condition];
  N02 -> N12[label=condition];
  N03 -> N12[label=consequence];
  N04 -> N05[label=mu];
  N04 -> N05[label=sigma];
  N05 -> N06[label=operand];
  N06 -> N11[label=consequence];
  N07 -> N09[label=mu];
  N08 -> N09[label=sigma];
  N08 -> N13[label=right];
  N09 -> N10[label=operand];
  N10 -> N11[label=alternative];
  N11 -> N12[label=alternative];
  N12 -> N13[label=left];
  N13 -> N14[label=operator];
}
"""
        self.assertEqual(observed_before.strip(), expected_before.strip())
        observed = to_dot(bmg, after_transform=True, label_edges=True)
        expected = """
digraph "graph" {
  N00[label=0.5];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=3.0];
  N04[label=3.0];
  N05[label=Normal];
  N06[label=Sample];
  N07[label=1.0];
  N08[label=2.0];
  N09[label=Normal];
  N10[label=Sample];
  N11[label=0.0];
  N12[label=if];
  N13[label=2.0];
  N14[label="*"];
  N15[label=Query];
  N00 -> N01[label=probability];
  N01 -> N02[label=operand];
  N02 -> N12[label=condition];
  N03 -> N05[label=mu];
  N04 -> N05[label=sigma];
  N05 -> N06[label=operand];
  N07 -> N09[label=mu];
  N08 -> N09[label=sigma];
  N09 -> N10[label=operand];
  N10 -> N12[label=alternative];
  N11 -> N12[label=consequence];
  N12 -> N14[label=left];
  N13 -> N14[label=right];
  N14 -> N15[label=operator];
}
"""
        self.assertEqual(observed.strip(), expected.strip())
