# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end test for 1D GMM with K > 2 number of components"""

import logging
import unittest

# Comments after imports suggest alternative comment style (for original tutorial)
import beanmachine.ppl as bm
import torch  # from torch import manual_seed, tensor
import torch.distributions as dist  # from torch.distributions import Bernoulli, Normal, Uniform
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor


# This makes the results deterministic and reproducible.

logging.getLogger("beanmachine").setLevel(50)
torch.manual_seed(42)

# Model


class GaussianMixtureModel(object):
    def __init__(self, k):
        self.K = k

    @bm.random_variable
    def alpha(self, k):
        return dist.Dirichlet(5 * torch.ones(k))

    @bm.random_variable
    def mu(self, c):
        return dist.Normal(0, 10)

    @bm.random_variable
    def sigma(self, c):
        return dist.Gamma(1, 10)

    @bm.random_variable
    def component(self, i):
        alpha = self.alpha(self.K)
        return dist.Categorical(alpha)

    @bm.random_variable
    def y(self, i):
        c = self.component(i)
        return dist.Normal(self.mu(c), self.sigma(c))


# Creating sample data

n = 6  # num observations
k = 4  # true number of clusters

gmm = GaussianMixtureModel(k=k)

ground_truth = {
    **{
        gmm.alpha(k): torch.ones(k) * 1.0 / k,
    },
    **{gmm.mu(i): tensor(i % 2).float() for i in range(k)},
    **{gmm.sigma(i): tensor(0.1) for i in range(k)},
    **{gmm.component(i): tensor(i % k).float() for i in range(n)},
}

# [Visualization code in tutorial skipped]

# Inference parameters
num_samples = (
    1  ###00 Sample size should not affect (the ability to find) compilation issues.
)

queries = (
    [gmm.alpha(gmm.K)]
    + [gmm.component(j) for j in range(n)]
    + [gmm.mu(i) for i in range(k)]
    + [gmm.sigma(i) for i in range(k)]
)

observations = {
    gmm.y(i): ground_truth[gmm.mu(ground_truth[gmm.component(i)].item())]
    for i in range(n)
}


class tutorialGMMwith1DimensionsAnd4Components(unittest.TestCase):
    def test_tutorial_GMM_with_1_dimensions_and_4_components(self) -> None:
        """Check BM and BMG inference both terminate"""

        self.maxDiff = None

        # Inference with BM

        torch.manual_seed(
            42
        )  # Note: Second time we seed. Could be a good tutorial style

        mh = bm.CompositionalInference()
        mh.infer(
            queries,
            observations,
            num_samples=num_samples,
            num_chains=1,
        )

        self.assertTrue(True, msg="We just want to check this point is reached")

    def test_tutorial_GMM_with_1_dimensions_and_4_components_to_dot_cpp_python(
        self,
    ) -> None:
        self.maxDiff = None
        observed = BMGInference().to_dot(queries, observations)
        expected = """digraph "graph" {
  N00[label="[5.0,5.0,5.0,5.0]"];
  N01[label=Dirichlet];
  N02[label=Sample];
  N03[label=Categorical];
  N04[label=Sample];
  N05[label=0.0];
  N06[label=10.0];
  N07[label=Normal];
  N08[label=Sample];
  N09[label=Sample];
  N10[label=Sample];
  N11[label=Sample];
  N12[label=1.0];
  N13[label=Gamma];
  N14[label=Sample];
  N15[label=Sample];
  N16[label=Sample];
  N17[label=Sample];
  N18[label=Choice];
  N19[label=Choice];
  N20[label=Normal];
  N21[label=Sample];
  N22[label="Observation 0.0"];
  N23[label=Sample];
  N24[label=Choice];
  N25[label=Choice];
  N26[label=Normal];
  N27[label=Sample];
  N28[label="Observation 1.0"];
  N29[label=Sample];
  N30[label=Choice];
  N31[label=Choice];
  N32[label=Normal];
  N33[label=Sample];
  N34[label="Observation 0.0"];
  N35[label=Sample];
  N36[label=Choice];
  N37[label=Choice];
  N38[label=Normal];
  N39[label=Sample];
  N40[label="Observation 1.0"];
  N41[label=Sample];
  N42[label=Choice];
  N43[label=Choice];
  N44[label=Normal];
  N45[label=Sample];
  N46[label="Observation 0.0"];
  N47[label=Sample];
  N48[label=Choice];
  N49[label=Choice];
  N50[label=Normal];
  N51[label=Sample];
  N52[label="Observation 1.0"];
  N53[label=Query];
  N54[label=Query];
  N55[label=Query];
  N56[label=Query];
  N57[label=Query];
  N58[label=Query];
  N59[label=Query];
  N60[label=Query];
  N61[label=Query];
  N62[label=Query];
  N63[label=Query];
  N64[label=Query];
  N65[label=Query];
  N66[label=Query];
  N67[label=Query];
  N00 -> N01;
  N01 -> N02;
  N02 -> N03;
  N02 -> N53;
  N03 -> N04;
  N03 -> N23;
  N03 -> N29;
  N03 -> N35;
  N03 -> N41;
  N03 -> N47;
  N04 -> N18;
  N04 -> N19;
  N04 -> N54;
  N05 -> N07;
  N06 -> N07;
  N06 -> N13;
  N07 -> N08;
  N07 -> N09;
  N07 -> N10;
  N07 -> N11;
  N08 -> N18;
  N08 -> N24;
  N08 -> N30;
  N08 -> N36;
  N08 -> N42;
  N08 -> N48;
  N08 -> N60;
  N09 -> N18;
  N09 -> N24;
  N09 -> N30;
  N09 -> N36;
  N09 -> N42;
  N09 -> N48;
  N09 -> N61;
  N10 -> N18;
  N10 -> N24;
  N10 -> N30;
  N10 -> N36;
  N10 -> N42;
  N10 -> N48;
  N10 -> N62;
  N11 -> N18;
  N11 -> N24;
  N11 -> N30;
  N11 -> N36;
  N11 -> N42;
  N11 -> N48;
  N11 -> N63;
  N12 -> N13;
  N13 -> N14;
  N13 -> N15;
  N13 -> N16;
  N13 -> N17;
  N14 -> N19;
  N14 -> N25;
  N14 -> N31;
  N14 -> N37;
  N14 -> N43;
  N14 -> N49;
  N14 -> N64;
  N15 -> N19;
  N15 -> N25;
  N15 -> N31;
  N15 -> N37;
  N15 -> N43;
  N15 -> N49;
  N15 -> N65;
  N16 -> N19;
  N16 -> N25;
  N16 -> N31;
  N16 -> N37;
  N16 -> N43;
  N16 -> N49;
  N16 -> N66;
  N17 -> N19;
  N17 -> N25;
  N17 -> N31;
  N17 -> N37;
  N17 -> N43;
  N17 -> N49;
  N17 -> N67;
  N18 -> N20;
  N19 -> N20;
  N20 -> N21;
  N21 -> N22;
  N23 -> N24;
  N23 -> N25;
  N23 -> N55;
  N24 -> N26;
  N25 -> N26;
  N26 -> N27;
  N27 -> N28;
  N29 -> N30;
  N29 -> N31;
  N29 -> N56;
  N30 -> N32;
  N31 -> N32;
  N32 -> N33;
  N33 -> N34;
  N35 -> N36;
  N35 -> N37;
  N35 -> N57;
  N36 -> N38;
  N37 -> N38;
  N38 -> N39;
  N39 -> N40;
  N41 -> N42;
  N41 -> N43;
  N41 -> N58;
  N42 -> N44;
  N43 -> N44;
  N44 -> N45;
  N45 -> N46;
  N47 -> N48;
  N47 -> N49;
  N47 -> N59;
  N48 -> N50;
  N49 -> N50;
  N50 -> N51;
  N51 -> N52;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_cpp(queries, observations)
        expected = """graph::Graph g;
Eigen::MatrixXd m0(4, 1)
m0 << 5.0, 5.0, 5.0, 5.0;
uint n0 = g.add_constant_pos_matrix(m0);
uint n1 = g.add_distribution(
  graph::DistributionType::DIRICHLET,
  graph::ValueType(
    graph::VariableType::COL_SIMPLEX_MATRIX,
    graph::AtomicType::PROBABILITY,
    4,
    1
  ),
  std::vector<uint>({n0}));
uint n2 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n3 = g.add_distribution(
  graph::DistributionType::CATEGORICAL,
  graph::AtomicType::NATURAL,
  std::vector<uint>({n2}));
uint n4 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n3}));
uint n5 = g.add_constant(0.0);
uint n6 = g.add_constant_pos_real(10.0);
uint n7 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n5, n6}));
uint n8 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n7}));
uint n9 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n7}));
uint n10 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n7}));
uint n11 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n7}));
uint n12 = g.add_constant_pos_real(1.0);
uint n13 = g.add_distribution(
  graph::DistributionType::GAMMA,
  graph::AtomicType::POS_REAL,
  std::vector<uint>({n12, n6}));
uint n14 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n13}));
uint n15 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n13}));
uint n16 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n13}));
uint n17 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n13}));
uint n18 = g.add_operator(
  graph::OperatorType::CHOICE,
  std::vector<uint>({n4, n8, n9, n10, n11}));
uint n19 = g.add_operator(
  graph::OperatorType::CHOICE,
  std::vector<uint>({n4, n14, n15, n16, n17}));
uint n20 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n18, n19}));
uint n21 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n20}));
g.observe([n21], 0.0);
uint n22 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n3}));
uint n23 = g.add_operator(
  graph::OperatorType::CHOICE,
  std::vector<uint>({n22, n8, n9, n10, n11}));
uint n24 = g.add_operator(
  graph::OperatorType::CHOICE,
  std::vector<uint>({n22, n14, n15, n16, n17}));
uint n25 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n23, n24}));
uint n26 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n25}));
g.observe([n26], 1.0);
uint n27 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n3}));
uint n28 = g.add_operator(
  graph::OperatorType::CHOICE,
  std::vector<uint>({n27, n8, n9, n10, n11}));
uint n29 = g.add_operator(
  graph::OperatorType::CHOICE,
  std::vector<uint>({n27, n14, n15, n16, n17}));
uint n30 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n28, n29}));
uint n31 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n30}));
g.observe([n31], 0.0);
uint n32 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n3}));
uint n33 = g.add_operator(
  graph::OperatorType::CHOICE,
  std::vector<uint>({n32, n8, n9, n10, n11}));
uint n34 = g.add_operator(
  graph::OperatorType::CHOICE,
  std::vector<uint>({n32, n14, n15, n16, n17}));
uint n35 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n33, n34}));
uint n36 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n35}));
g.observe([n36], 1.0);
uint n37 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n3}));
uint n38 = g.add_operator(
  graph::OperatorType::CHOICE,
  std::vector<uint>({n37, n8, n9, n10, n11}));
uint n39 = g.add_operator(
  graph::OperatorType::CHOICE,
  std::vector<uint>({n37, n14, n15, n16, n17}));
uint n40 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n38, n39}));
uint n41 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n40}));
g.observe([n41], 0.0);
uint n42 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n3}));
uint n43 = g.add_operator(
  graph::OperatorType::CHOICE,
  std::vector<uint>({n42, n8, n9, n10, n11}));
uint n44 = g.add_operator(
  graph::OperatorType::CHOICE,
  std::vector<uint>({n42, n14, n15, n16, n17}));
uint n45 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n43, n44}));
uint n46 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n45}));
g.observe([n46], 1.0);
uint q0 = g.query(n2);
uint q1 = g.query(n4);
uint q2 = g.query(n22);
uint q3 = g.query(n27);
uint q4 = g.query(n32);
uint q5 = g.query(n37);
uint q6 = g.query(n42);
uint q7 = g.query(n8);
uint q8 = g.query(n9);
uint q9 = g.query(n10);
uint q10 = g.query(n11);
uint q11 = g.query(n14);
uint q12 = g.query(n15);
uint q13 = g.query(n16);
uint q14 = g.query(n17);
        """
        self.assertEqual(expected.strip(), observed.strip())
