# Copyright (c) Facebook, Inc. and its affiliates.
"""End-to-end test of realistic coin flip model"""
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli, Beta


@bm.random_variable
def beta():
    return Beta(2.0, 2.0)


@bm.random_variable
def flip(n):
    return Bernoulli(beta())


class CoinFlipTest(unittest.TestCase):
    def test_coin_flip_inference(self) -> None:
        """test_inference from coin_flip_test.py"""

        # We've got a prior on the coin of Beta(2,2), so it is most
        # likely to be actually fair, but still with some probability
        # of being unfair in either direction.
        #
        # We flip the coin four times and get heads 25% of the time,
        # so this is some evidence that the true fairness of the coin is
        # closer to 25% than 50%.
        #
        # We sample 1000 times from the posterior and take the average;
        # it should come out that the true fairness is now most likely
        # to be around 37%.

        self.maxDiff = None
        queries = [beta()]
        observations = {
            flip(0): tensor(0.0),
            flip(1): tensor(0.0),
            flip(2): tensor(1.0),
            flip(3): tensor(0.0),
        }
        num_samples = 1000
        inference = BMGInference()
        mcsamples = inference.infer(queries, observations, num_samples)
        samples = mcsamples[beta()]
        observed = samples.mean()
        expected = 0.37
        self.assertAlmostEqual(first=observed, second=expected, delta=0.05)

    def test_coin_flip_to_dot_cpp_python(self) -> None:
        self.maxDiff = None
        queries = [beta()]
        observations = {
            flip(0): tensor(0.0),
            flip(1): tensor(0.0),
            flip(2): tensor(1.0),
            flip(3): tensor(0.0),
        }
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N00[label=2.0];
  N01[label=Beta];
  N02[label=Sample];
  N03[label=Bernoulli];
  N04[label=Sample];
  N05[label="Observation False"];
  N06[label=Sample];
  N07[label="Observation False"];
  N08[label=Sample];
  N09[label="Observation True"];
  N10[label=Sample];
  N11[label="Observation False"];
  N12[label=Query];
  N00 -> N01;
  N00 -> N01;
  N01 -> N02;
  N02 -> N03;
  N02 -> N12;
  N03 -> N04;
  N03 -> N06;
  N03 -> N08;
  N03 -> N10;
  N04 -> N05;
  N06 -> N07;
  N08 -> N09;
  N10 -> N11;
}
        """
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_cpp(queries, observations)
        expected = """
graph::Graph g;
uint n0 = g.add_constant_pos_real(2.0);
uint n1 = g.add_distribution(
  graph::DistributionType::BETA,
  graph::AtomicType::PROBABILITY,
  std::vector<uint>({n0, n0}));
uint n2 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n3 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n2}));
uint n4 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n3}));
g.observe([n4], false);
uint n5 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n3}));
g.observe([n5], false);
uint n6 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n3}));
g.observe([n6], true);
uint n7 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n3}));
g.observe([n7], false);
uint q0 = g.query(n2);"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_python(queries, observations)
        expected = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_pos_real(2.0)
n1 = g.add_distribution(
  graph.DistributionType.BETA,
  graph.AtomicType.PROBABILITY,
  [n0, n0],
)
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n3 = g.add_distribution(
  graph.DistributionType.BERNOULLI,
  graph.AtomicType.BOOLEAN,
  [n2],
)
n4 = g.add_operator(graph.OperatorType.SAMPLE, [n3])
g.observe(n4, False)
n5 = g.add_operator(graph.OperatorType.SAMPLE, [n3])
g.observe(n5, False)
n6 = g.add_operator(graph.OperatorType.SAMPLE, [n3])
g.observe(n6, True)
n7 = g.add_operator(graph.OperatorType.SAMPLE, [n3])
g.observe(n7, False)
q0 = g.query(n2)"""
        self.assertEqual(expected.strip(), observed.strip())
