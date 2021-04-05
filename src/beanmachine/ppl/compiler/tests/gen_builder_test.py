# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for gen_builder.py"""
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.gen_builder import generate_builder
from torch.distributions import Normal


@bm.random_variable
def norm(x):
    return Normal(0.0, 1.0)


@bm.functional
def norm_sum():
    return norm(1) + norm(2) + norm(3) + norm(4)


class GenerateBuilderTest(unittest.TestCase):
    def test_generate_builder_1(self) -> None:
        self.maxDiff = None
        bmg = BMGraphBuilder()
        bmg.accumulate_graph([norm_sum()], {})
        observed = generate_builder(bmg)
        expected = """
import beanmachine.ppl.compiler.bmg_nodes as bn
import torch
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from torch import tensor

bmg = BMGraphBuilder()
n0 = bmg.add_node(bn.ConstantTensorNode(tensor(0.)))
n1 = bmg.add_node(bn.ConstantTensorNode(tensor(1.)))
n2 = bmg.add_node(bn.NormalNode(n0, n1))
n3 = bmg.add_node(bn.SampleNode(n2))
n4 = bmg.add_node(bn.SampleNode(n2))
n5 = bmg.add_node(bn.AdditionNode(n3, n4))
n6 = bmg.add_node(bn.SampleNode(n2))
n7 = bmg.add_node(bn.AdditionNode(n5, n6))
n8 = bmg.add_node(bn.SampleNode(n2))
n9 = bmg.add_node(bn.AdditionNode(n7, n8))
n10 = bmg.add_node(bn.Query(n9))"""
        self.assertEqual(expected.strip(), observed.strip())
