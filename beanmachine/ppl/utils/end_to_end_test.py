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
uint n0 = g.add_constant_probability(0.5);
uint n1 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n0}));
uint n2 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
"""

expected_bmg_1 = """
Node 0 type 1 parents [ ] children [ 1 ] probability value 0.5
Node 1 type 2 parents [ 0 ] children [ 2 ] unknown value
Node 2 type 3 parents [ 1 ] children [ ] boolean value 0
"""

expected_python_1 = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_probability(0.5)
n1 = g.add_distribution(
  graph.DistributionType.BERNOULLI,
  graph.AtomicType.BOOLEAN,
  [n0])
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
"""


class EndToEndTest(unittest.TestCase):
    def test_to_cpp(self) -> None:
        """test_to_cpp from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_cpp(source_1)
        self.assertEqual(observed.strip(), expected_cpp_1.strip())

    def test_to_bmg(self) -> None:
        """test_to_bmg from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_bmg(source_1).to_string()
        self.assertEqual(tidy(observed), tidy(expected_bmg_1))

    def test_to_python(self) -> None:
        """test_to_python from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_python(source_1)
        self.assertEqual(observed.strip(), expected_python_1.strip())
