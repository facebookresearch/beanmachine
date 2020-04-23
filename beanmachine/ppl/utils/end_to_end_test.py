# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bm_to_bmg.py"""
import unittest

from beanmachine.ppl.utils.bm_to_bmg import to_bmg, to_cpp, to_python


def tidy(s: str) -> str:
    return "\n".join(c.strip() for c in s.strip().split("\n")).strip()


source_1 = """
from beanmachine.ppl.model.statistical_model import sample
import torch
from torch import tensor
from torch.distributions.bernoulli import Bernoulli

@sample
def flip():
  return Bernoulli(tensor(0.5))
"""

expected_cpp_1 = """
graph::Graph g;
uint n3 = g.add_constant_probability(0.5);
uint n4 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n3}));
uint n2 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n4}));
"""

expected_bmg_1 = """TODO"""

expected_python_1 = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n3 = g.add_constant_probability(0.5)
n4 = g.add_distribution(
  graph.DistributionType.BERNOULLI,
  graph.AtomicType.BOOLEAN,
  [n3])
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n4])
"""


class EndToEndTest(unittest.TestCase):
    def test_to_cpp(self) -> None:
        """test_to_cpp from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_cpp(source_1)
        self.assertEqual(observed.strip(), expected_cpp_1.strip())

    def disabled_test_to_bmg(self) -> None:
        """test_to_bmg from end_to_end_test.py"""
        # TODO: This test is disabled because of an undiagnosed crash in
        # TODO: to_string.
        self.maxDiff = None
        observed = to_bmg(source_1).to_string()
        self.assertEqual(tidy(observed), tidy(expected_bmg_1))

    def test_to_python(self) -> None:
        """test_to_python from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_python(source_1)
        self.assertEqual(observed.strip(), expected_python_1.strip())
