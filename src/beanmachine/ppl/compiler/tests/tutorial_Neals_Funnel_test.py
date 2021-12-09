# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end test for tutorial on Neal's Funnel"""
# This file is a manual replica of the Bento tutorial with the same name
# This is a block for Beanstalk OSS readiness
# TODO: Check imports for conistency

import logging
import math
import unittest

import beanmachine.ppl as bm
import torch  # from torch import manual_seed, tensor
import torch.distributions as dist  # from torch.distributions import Bernoulli, Normal, Uniform
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor


# This makes the results deterministic and reproducible.

logging.getLogger("beanmachine").setLevel(50)
torch.manual_seed(11)

# Model
def normal_log_prob(mu, sigma, x):
    z = (x - mu) / sigma
    return (-1.0 / 2.0) * math.log(2.0 * math.pi) - (z ** 2.0 / 2.0)


@bm.random_variable
def z():
    """
    An uninformative (flat) prior for z.
    """
    # TODO(tingley): Replace with Flat once it's part of the framework.
    return dist.Normal(0, 10000)


@bm.random_variable
def x():
    """
    An uninformative (flat) prior for x.
    """
    # TODO(tingley): Replace with Flat once it's part of the framework.
    return dist.Normal(0, 10000)


@bm.random_variable
def neals_funnel_coin_flip():
    """
    Flip a "coin", which is heads with probability equal to the probability
    of drawing z and x from the true Neal's funnel posterior.
    """
    return dist.Bernoulli(
        (
            normal_log_prob(0.0, 3.0, z())
            + normal_log_prob(0.0, (z() / 2.0).exp(), x())
        ).exp()
    )


# Inference parameters

num_samples = 1  ###000
num_chains = 4

observations = {neals_funnel_coin_flip(): tensor(1.0)}

queries = [z(), x()]


class tutorialNealsFunnelTest(unittest.TestCase):
    def test_tutorial_Neals_Funnel(self) -> None:
        """Check BM and BMG inference both terminate"""

        self.maxDiff = None

        # Inference with BM

        # Note: No explicit seed here (in original tutorial model). Should we add one?
        nmc = bm.SingleSiteNewtonianMonteCarlo()
        _ = nmc.infer(
            queries=queries,
            observations=observations,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        hmc = bm.SingleSiteHamiltonianMonteCarlo(
            trajectory_length=0.1, initial_step_size=0.01
        )
        _ = hmc.infer(
            queries=queries,
            observations=observations,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        ghmc = bm.CompositionalInference(
            {
                (z, x): bm.SingleSiteHamiltonianMonteCarlo(
                    trajectory_length=0.1, initial_step_size=0.01
                ),
            }
        )
        ghmc.infer(
            queries=queries,
            observations=observations,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        bmg = BMGInference()
        _ = bmg.infer(
            queries=queries,
            observations=observations,
            num_samples=num_samples,
            num_chains=1,  # TODO[Walid]: 1 should be num_chains
        )

        self.assertTrue(True, msg="We just want to check this point is reached")

    def test_tutorial_Neals_Funnel_to_dot_cpp_python(
        self,
    ) -> None:
        self.maxDiff = None
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=10000.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=Sample];
  N05[label=-0.9189385332046727];
  N06[label=0.3333333333333333];
  N07[label="*"];
  N08[label=2.0];
  N09[label="**"];
  N10[label=0.5];
  N11[label="*"];
  N12[label="-"];
  N13[label="*"];
  N14[label=Exp];
  N15[label=-1.0];
  N16[label="**"];
  N17[label=ToReal];
  N18[label="*"];
  N19[label="**"];
  N20[label="*"];
  N21[label="-"];
  N22[label="+"];
  N23[label=Exp];
  N24[label=ToProb];
  N25[label=Bernoulli];
  N26[label=Sample];
  N27[label="Observation True"];
  N28[label=Query];
  N29[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N02 -> N04;
  N03 -> N07;
  N03 -> N13;
  N03 -> N28;
  N04 -> N18;
  N04 -> N29;
  N05 -> N22;
  N05 -> N22;
  N06 -> N07;
  N07 -> N09;
  N08 -> N09;
  N08 -> N19;
  N09 -> N11;
  N10 -> N11;
  N10 -> N13;
  N10 -> N20;
  N11 -> N12;
  N12 -> N22;
  N13 -> N14;
  N14 -> N16;
  N15 -> N16;
  N16 -> N17;
  N17 -> N18;
  N18 -> N19;
  N19 -> N20;
  N20 -> N21;
  N21 -> N22;
  N22 -> N23;
  N23 -> N24;
  N24 -> N25;
  N25 -> N26;
  N26 -> N27;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_cpp(queries, observations)
        expected = """
graph::Graph g;
uint n0 = g.add_constant(0.0);
uint n1 = g.add_constant_pos_real(10000.0);
uint n2 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n0, n1}));
uint n3 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n2}));
uint n4 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n2}));
uint n5 = g.add_constant(-0.9189385332046727);
uint n6 = g.add_constant(0.3333333333333333);
uint n7 = g.add_operator(
  graph::OperatorType::MULTIPLY, std::vector<uint>({n3, n6}));
uint n8 = g.add_constant_pos_real(2.0);
uint n9 = g.add_operator(
  graph::OperatorType::POW, std::vector<uint>({n7, n8}));
uint n10 = g.add_constant(0.5);
uint n11 = g.add_operator(
  graph::OperatorType::MULTIPLY, std::vector<uint>({n9, n10}));
uint n12 = g.add_operator(
  graph::OperatorType::NEGATE, std::vector<uint>({n11}));
uint n13 = g.add_operator(
  graph::OperatorType::MULTIPLY, std::vector<uint>({n3, n10}));
uint n14 = g.add_operator(
  graph::OperatorType::EXP, std::vector<uint>({n13}));
uint n15 = g.add_constant(-1.0);
uint n16 = g.add_operator(
  graph::OperatorType::POW, std::vector<uint>({n14, n15}));
uint n17 = g.add_operator(
  graph::OperatorType::TO_REAL, std::vector<uint>({n16}));
uint n18 = g.add_operator(
  graph::OperatorType::MULTIPLY, std::vector<uint>({n4, n17}));
uint n19 = g.add_operator(
  graph::OperatorType::POW, std::vector<uint>({n18, n8}));
uint n20 = g.add_operator(
  graph::OperatorType::MULTIPLY, std::vector<uint>({n19, n10}));
uint n21 = g.add_operator(
  graph::OperatorType::NEGATE, std::vector<uint>({n20}));
uint n22 = g.add_operator(
  graph::OperatorType::ADD,
  std::vector<uint>({n5, n12, n5, n21}));
uint n23 = g.add_operator(
  graph::OperatorType::EXP, std::vector<uint>({n22}));
uint n24 = g.add_operator(
  graph::OperatorType::TO_PROBABILITY, std::vector<uint>({n23}));
uint n25 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n24}));
uint n26 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n25}));
g.observe([n26], true);
uint q0 = g.query(n3);
uint q1 = g.query(n4);
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_python(queries, observations)
        expected = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_real(0.0)
n1 = g.add_constant_pos_real(10000.0)
n2 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n0, n1],
)
n3 = g.add_operator(graph.OperatorType.SAMPLE, [n2])
n4 = g.add_operator(graph.OperatorType.SAMPLE, [n2])
n5 = g.add_constant_real(-0.9189385332046727)
n6 = g.add_constant_real(0.3333333333333333)
n7 = g.add_operator(graph.OperatorType.MULTIPLY, [n3, n6])
n8 = g.add_constant_pos_real(2.0)
n9 = g.add_operator(graph.OperatorType.POW, [n7, n8])
n10 = g.add_constant_real(0.5)
n11 = g.add_operator(graph.OperatorType.MULTIPLY, [n9, n10])
n12 = g.add_operator(graph.OperatorType.NEGATE, [n11])
n13 = g.add_operator(graph.OperatorType.MULTIPLY, [n3, n10])
n14 = g.add_operator(graph.OperatorType.EXP, [n13])
n15 = g.add_constant_real(-1.0)
n16 = g.add_operator(graph.OperatorType.POW, [n14, n15])
n17 = g.add_operator(graph.OperatorType.TO_REAL, [n16])
n18 = g.add_operator(graph.OperatorType.MULTIPLY, [n4, n17])
n19 = g.add_operator(graph.OperatorType.POW, [n18, n8])
n20 = g.add_operator(graph.OperatorType.MULTIPLY, [n19, n10])
n21 = g.add_operator(graph.OperatorType.NEGATE, [n20])
n22 = g.add_operator(
  graph.OperatorType.ADD,
  [n5, n12, n5, n21],
)
n23 = g.add_operator(graph.OperatorType.EXP, [n22])
n24 = g.add_operator(graph.OperatorType.TO_PROBABILITY, [n23])
n25 = g.add_distribution(
  graph.DistributionType.BERNOULLI,
  graph.AtomicType.BOOLEAN,
  [n24],
)
n26 = g.add_operator(graph.OperatorType.SAMPLE, [n25])
g.observe(n26, True)
q0 = g.query(n3)
q1 = g.query(n4)
"""
        self.assertEqual(expected.strip(), observed.strip())
