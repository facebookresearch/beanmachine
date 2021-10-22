# Copyright (c) Facebook, Inc. and its affiliates.
"""End-to-end test for an example use of the normal distribution"""

import logging
import unittest

import beanmachine.ppl as bm
import torch  # from torch import manual_seed, tensor
import torch.distributions as dist  # from torch.distributions import Bernoulli, Normal, Uniform
from beanmachine.ppl.inference.bmg_inference import BMGInference


# TODO: Check imports for consistency


# This makes the results deterministic and reproducible.

logging.getLogger("beanmachine").setLevel(50)
torch.manual_seed(12)

# Model


@bm.random_variable
def x():
    """
    A random variable drawn from a half normal distribution
    """
    return dist.HalfNormal(1000)


num_samples = (
    2  ###000 - Sample size reduced since it should not affect compilation issues
)
num_chains = 4

observations = {}  ### This means we will just get the distribution as declared

queries = [x()]


class distributionHalfNormalTest(unittest.TestCase):
    def test_distribution_half_normal_e2e(self) -> None:
        """Check BM and BMG inference both terminate"""

        self.maxDiff = None

        # Inference with BM

        # Note: No explicit seed here (in original tutorial model). Should we add one?
        amh = bm.SingleSiteAncestralMetropolisHastings()  # Added local binding
        bm_samples = amh.infer(
            queries=queries,
            observations=observations,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        self.assertTrue(
            bm_samples.get_num_samples() == num_samples,
            msg="Got wrong number of samples back from BM inference",
        )

        # Inference with BMG
        bmg_samples = BMGInference().infer(
            queries=queries,
            observations=observations,
            num_samples=num_samples,
            num_chains=1,  # TODO[Walid]: 1 should be replaced by num_chains
        )

        self.assertTrue(
            bmg_samples.get_num_samples() == num_samples,
            msg="Got wrong number of samples back from BMG inference",
        )

    def test_distribution_half_normal_to_dot_cpp_python(self,) -> None:
        self.maxDiff = None
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N0[label=1000.0];
  N1[label=HalfNormal];
  N2[label=Sample];
  N3[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
}
        """
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_cpp(queries, observations)
        expected = """
graph::Graph g;
uint n0 = g.add_constant_pos_real(1000.0);
uint n1 = g.add_distribution(
  graph::DistributionType::HALF_NORMAL,
  graph::AtomicType::POS_REAL,
  std::vector<uint>({n0}));
uint n2 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint q0 = g.query(n2);
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_python(queries, observations)
        expected = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_pos_real(1000.0)
n1 = g.add_distribution(
  graph.DistributionType.HALF_NORMAL,
  graph.AtomicType.POS_REAL,
  [n0],
)
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
q0 = g.query(n2)
"""
        self.assertEqual(expected.strip(), observed.strip())
