# Copyright (c) Facebook, Inc. and its affiliates.
"""End-to-end test for tutorial on GMM with Poisson number of components"""
# This file is a manual replica of the Bento tutorial with the same name
# TODO: The disabled test generates the following error:
# E       TypeError: Distribution 'Poisson' is not supported by Bean Machine Graph.
# This will need to be fixed for OSS readiness task

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
    @bm.random_variable
    def mu(self, c):
        return dist.Normal(0.0, 10.0)

    @bm.random_variable
    def sigma(self, c):
        return dist.Gamma(1, 1)

    @bm.random_variable
    def component(self, i):
        return dist.Bernoulli(probs=0.5)

    @bm.random_variable
    def y(self, i):
        c = self.component(i)
        return dist.Normal(self.mu(c), self.sigma(c))


# Creating sample data

n = 12  # num observations
k = 2  # true number of clusters

gmm = GaussianMixtureModel()

ground_truth = {
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
    [gmm.component(j) for j in range(n)]
    + [gmm.mu(i) for i in range(k)]
    + [gmm.sigma(i) for i in range(k)]
)

observations = {
    gmm.y(i): ground_truth[gmm.mu(ground_truth[gmm.component(i)].item())]
    for i in range(n)
}


class tutorialGMM1Dimension2Components(unittest.TestCase):
    def test_tutorial_GMM_1_dimension_2_components(self) -> None:
        """Check BM and BMG inference both terminate"""

        self.maxDiff = None

        # Inference with BM

        torch.manual_seed(
            42
        )  # Note: Second time we seed. Could be a good tutorial style

        mh = bm.CompositionalInference()
        mh.infer(
            queries, observations, num_samples=num_samples, num_chains=1,
        )

        bmg = BMGInference()
        bmg.infer(
            queries=queries,
            observations=observations,
            num_samples=num_samples,
            num_chains=1,
        )

        self.assertTrue(True, msg="We just want to check this point is reached")

    def test_tutorial_GMM_1_dimension_2_components_to_dot_cpp_python(self,) -> None:
        self.maxDiff = None
        observed = BMGInference().to_dot(queries, observations)
        expected = """digraph "graph" {
  N00[label=0.5];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=0.0];
  N04[label=10.0];
  N05[label=Normal];
  N06[label=Sample];
  N07[label=Sample];
  N08[label=1.0];
  N09[label=Gamma];
  N10[label=Sample];
  N11[label=Sample];
  N12[label=if];
  N13[label=if];
  N14[label=Normal];
  N15[label=Sample];
  N16[label="Observation 0.0"];
  N17[label=Sample];
  N18[label=if];
  N19[label=if];
  N20[label=Normal];
  N21[label=Sample];
  N22[label="Observation 1.0"];
  N23[label=Sample];
  N24[label=if];
  N25[label=if];
  N26[label=Normal];
  N27[label=Sample];
  N28[label="Observation 0.0"];
  N29[label=Sample];
  N30[label=if];
  N31[label=if];
  N32[label=Normal];
  N33[label=Sample];
  N34[label="Observation 1.0"];
  N35[label=Sample];
  N36[label=if];
  N37[label=if];
  N38[label=Normal];
  N39[label=Sample];
  N40[label="Observation 0.0"];
  N41[label=Sample];
  N42[label=if];
  N43[label=if];
  N44[label=Normal];
  N45[label=Sample];
  N46[label="Observation 1.0"];
  N47[label=Sample];
  N48[label=if];
  N49[label=if];
  N50[label=Normal];
  N51[label=Sample];
  N52[label="Observation 0.0"];
  N53[label=Sample];
  N54[label=if];
  N55[label=if];
  N56[label=Normal];
  N57[label=Sample];
  N58[label="Observation 1.0"];
  N59[label=Sample];
  N60[label=if];
  N61[label=if];
  N62[label=Normal];
  N63[label=Sample];
  N64[label="Observation 0.0"];
  N65[label=Sample];
  N66[label=if];
  N67[label=if];
  N68[label=Normal];
  N69[label=Sample];
  N70[label="Observation 1.0"];
  N71[label=Sample];
  N72[label=if];
  N73[label=if];
  N74[label=Normal];
  N75[label=Sample];
  N76[label="Observation 0.0"];
  N77[label=Sample];
  N78[label=if];
  N79[label=if];
  N80[label=Normal];
  N81[label=Sample];
  N82[label="Observation 1.0"];
  N83[label=Query];
  N84[label=Query];
  N85[label=Query];
  N86[label=Query];
  N87[label=Query];
  N88[label=Query];
  N89[label=Query];
  N90[label=Query];
  N91[label=Query];
  N92[label=Query];
  N93[label=Query];
  N94[label=Query];
  N95[label=Query];
  N96[label=Query];
  N97[label=Query];
  N98[label=Query];
  N00 -> N01;
  N01 -> N02;
  N01 -> N17;
  N01 -> N23;
  N01 -> N29;
  N01 -> N35;
  N01 -> N41;
  N01 -> N47;
  N01 -> N53;
  N01 -> N59;
  N01 -> N65;
  N01 -> N71;
  N01 -> N77;
  N02 -> N12;
  N02 -> N13;
  N02 -> N83;
  N03 -> N05;
  N04 -> N05;
  N05 -> N06;
  N05 -> N07;
  N06 -> N12;
  N06 -> N18;
  N06 -> N24;
  N06 -> N30;
  N06 -> N36;
  N06 -> N42;
  N06 -> N48;
  N06 -> N54;
  N06 -> N60;
  N06 -> N66;
  N06 -> N72;
  N06 -> N78;
  N06 -> N95;
  N07 -> N12;
  N07 -> N18;
  N07 -> N24;
  N07 -> N30;
  N07 -> N36;
  N07 -> N42;
  N07 -> N48;
  N07 -> N54;
  N07 -> N60;
  N07 -> N66;
  N07 -> N72;
  N07 -> N78;
  N07 -> N96;
  N08 -> N09;
  N08 -> N09;
  N09 -> N10;
  N09 -> N11;
  N10 -> N13;
  N10 -> N19;
  N10 -> N25;
  N10 -> N31;
  N10 -> N37;
  N10 -> N43;
  N10 -> N49;
  N10 -> N55;
  N10 -> N61;
  N10 -> N67;
  N10 -> N73;
  N10 -> N79;
  N10 -> N97;
  N11 -> N13;
  N11 -> N19;
  N11 -> N25;
  N11 -> N31;
  N11 -> N37;
  N11 -> N43;
  N11 -> N49;
  N11 -> N55;
  N11 -> N61;
  N11 -> N67;
  N11 -> N73;
  N11 -> N79;
  N11 -> N98;
  N12 -> N14;
  N13 -> N14;
  N14 -> N15;
  N15 -> N16;
  N17 -> N18;
  N17 -> N19;
  N17 -> N84;
  N18 -> N20;
  N19 -> N20;
  N20 -> N21;
  N21 -> N22;
  N23 -> N24;
  N23 -> N25;
  N23 -> N85;
  N24 -> N26;
  N25 -> N26;
  N26 -> N27;
  N27 -> N28;
  N29 -> N30;
  N29 -> N31;
  N29 -> N86;
  N30 -> N32;
  N31 -> N32;
  N32 -> N33;
  N33 -> N34;
  N35 -> N36;
  N35 -> N37;
  N35 -> N87;
  N36 -> N38;
  N37 -> N38;
  N38 -> N39;
  N39 -> N40;
  N41 -> N42;
  N41 -> N43;
  N41 -> N88;
  N42 -> N44;
  N43 -> N44;
  N44 -> N45;
  N45 -> N46;
  N47 -> N48;
  N47 -> N49;
  N47 -> N89;
  N48 -> N50;
  N49 -> N50;
  N50 -> N51;
  N51 -> N52;
  N53 -> N54;
  N53 -> N55;
  N53 -> N90;
  N54 -> N56;
  N55 -> N56;
  N56 -> N57;
  N57 -> N58;
  N59 -> N60;
  N59 -> N61;
  N59 -> N91;
  N60 -> N62;
  N61 -> N62;
  N62 -> N63;
  N63 -> N64;
  N65 -> N66;
  N65 -> N67;
  N65 -> N92;
  N66 -> N68;
  N67 -> N68;
  N68 -> N69;
  N69 -> N70;
  N71 -> N72;
  N71 -> N73;
  N71 -> N93;
  N72 -> N74;
  N73 -> N74;
  N74 -> N75;
  N75 -> N76;
  N77 -> N78;
  N77 -> N79;
  N77 -> N94;
  N78 -> N80;
  N79 -> N80;
  N80 -> N81;
  N81 -> N82;
}
        """
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_cpp(queries, observations)
        expected = """graph::Graph g;
uint n0 = g.add_constant_probability(0.5);
uint n1 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n0}));
uint n2 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n3 = g.add_constant(0.0);
uint n4 = g.add_constant_pos_real(10.0);
uint n5 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n3, n4}));
uint n6 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n5}));
uint n7 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n5}));
uint n8 = g.add_constant_pos_real(1.0);
uint n9 = g.add_distribution(
  graph::DistributionType::GAMMA,
  graph::AtomicType::POS_REAL,
  std::vector<uint>({n8, n8}));
uint n10 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n9}));
uint n11 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n9}));
uint n12 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n2, n7, n6}));
uint n13 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n2, n11, n10}));
uint n14 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n12, n13}));
uint n15 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n14}));
g.observe([n15], 0.0);
uint n16 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n17 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n16, n7, n6}));
uint n18 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n16, n11, n10}));
uint n19 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n17, n18}));
uint n20 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n19}));
g.observe([n20], 1.0);
uint n21 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n22 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n21, n7, n6}));
uint n23 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n21, n11, n10}));
uint n24 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n22, n23}));
uint n25 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n24}));
g.observe([n25], 0.0);
uint n26 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n27 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n26, n7, n6}));
uint n28 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n26, n11, n10}));
uint n29 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n27, n28}));
uint n30 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n29}));
g.observe([n30], 1.0);
uint n31 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n32 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n31, n7, n6}));
uint n33 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n31, n11, n10}));
uint n34 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n32, n33}));
uint n35 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n34}));
g.observe([n35], 0.0);
uint n36 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n37 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n36, n7, n6}));
uint n38 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n36, n11, n10}));
uint n39 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n37, n38}));
uint n40 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n39}));
g.observe([n40], 1.0);
uint n41 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n42 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n41, n7, n6}));
uint n43 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n41, n11, n10}));
uint n44 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n42, n43}));
uint n45 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n44}));
g.observe([n45], 0.0);
uint n46 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n47 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n46, n7, n6}));
uint n48 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n46, n11, n10}));
uint n49 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n47, n48}));
uint n50 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n49}));
g.observe([n50], 1.0);
uint n51 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n52 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n51, n7, n6}));
uint n53 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n51, n11, n10}));
uint n54 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n52, n53}));
uint n55 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n54}));
g.observe([n55], 0.0);
uint n56 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n57 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n56, n7, n6}));
uint n58 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n56, n11, n10}));
uint n59 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n57, n58}));
uint n60 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n59}));
g.observe([n60], 1.0);
uint n61 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n62 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n61, n7, n6}));
uint n63 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n61, n11, n10}));
uint n64 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n62, n63}));
uint n65 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n64}));
g.observe([n65], 0.0);
uint n66 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n67 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n66, n7, n6}));
uint n68 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n66, n11, n10}));
uint n69 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n67, n68}));
uint n70 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n69}));
g.observe([n70], 1.0);
uint q0 = g.query(n2);
uint q1 = g.query(n16);
uint q2 = g.query(n21);
uint q3 = g.query(n26);
uint q4 = g.query(n31);
uint q5 = g.query(n36);
uint q6 = g.query(n41);
uint q7 = g.query(n46);
uint q8 = g.query(n51);
uint q9 = g.query(n56);
uint q10 = g.query(n61);
uint q11 = g.query(n66);
uint q12 = g.query(n6);
uint q13 = g.query(n7);
uint q14 = g.query(n10);
uint q15 = g.query(n11);
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_python(queries, observations)
        expected = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_probability(0.5)
n1 = g.add_distribution(
  graph.DistributionType.BERNOULLI,
  graph.AtomicType.BOOLEAN,
  [n0],
)
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n3 = g.add_constant_real(0.0)
n4 = g.add_constant_pos_real(10.0)
n5 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n3, n4],
)
n6 = g.add_operator(graph.OperatorType.SAMPLE, [n5])
n7 = g.add_operator(graph.OperatorType.SAMPLE, [n5])
n8 = g.add_constant_pos_real(1.0)
n9 = g.add_distribution(
  graph.DistributionType.GAMMA,
  graph.AtomicType.POS_REAL,
  [n8, n8],
)
n10 = g.add_operator(graph.OperatorType.SAMPLE, [n9])
n11 = g.add_operator(graph.OperatorType.SAMPLE, [n9])
n12 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n2, n7, n6],
)
n13 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n2, n11, n10],
)
n14 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n12, n13],
)
n15 = g.add_operator(graph.OperatorType.SAMPLE, [n14])
g.observe(n15, 0.0)
n16 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n17 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n16, n7, n6],
)
n18 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n16, n11, n10],
)
n19 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n17, n18],
)
n20 = g.add_operator(graph.OperatorType.SAMPLE, [n19])
g.observe(n20, 1.0)
n21 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n22 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n21, n7, n6],
)
n23 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n21, n11, n10],
)
n24 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n22, n23],
)
n25 = g.add_operator(graph.OperatorType.SAMPLE, [n24])
g.observe(n25, 0.0)
n26 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n27 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n26, n7, n6],
)
n28 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n26, n11, n10],
)
n29 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n27, n28],
)
n30 = g.add_operator(graph.OperatorType.SAMPLE, [n29])
g.observe(n30, 1.0)
n31 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n32 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n31, n7, n6],
)
n33 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n31, n11, n10],
)
n34 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n32, n33],
)
n35 = g.add_operator(graph.OperatorType.SAMPLE, [n34])
g.observe(n35, 0.0)
n36 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n37 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n36, n7, n6],
)
n38 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n36, n11, n10],
)
n39 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n37, n38],
)
n40 = g.add_operator(graph.OperatorType.SAMPLE, [n39])
g.observe(n40, 1.0)
n41 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n42 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n41, n7, n6],
)
n43 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n41, n11, n10],
)
n44 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n42, n43],
)
n45 = g.add_operator(graph.OperatorType.SAMPLE, [n44])
g.observe(n45, 0.0)
n46 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n47 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n46, n7, n6],
)
n48 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n46, n11, n10],
)
n49 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n47, n48],
)
n50 = g.add_operator(graph.OperatorType.SAMPLE, [n49])
g.observe(n50, 1.0)
n51 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n52 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n51, n7, n6],
)
n53 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n51, n11, n10],
)
n54 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n52, n53],
)
n55 = g.add_operator(graph.OperatorType.SAMPLE, [n54])
g.observe(n55, 0.0)
n56 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n57 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n56, n7, n6],
)
n58 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n56, n11, n10],
)
n59 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n57, n58],
)
n60 = g.add_operator(graph.OperatorType.SAMPLE, [n59])
g.observe(n60, 1.0)
n61 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n62 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n61, n7, n6],
)
n63 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n61, n11, n10],
)
n64 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n62, n63],
)
n65 = g.add_operator(graph.OperatorType.SAMPLE, [n64])
g.observe(n65, 0.0)
n66 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n67 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n66, n7, n6],
)
n68 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n66, n11, n10],
)
n69 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n67, n68],
)
n70 = g.add_operator(graph.OperatorType.SAMPLE, [n69])
g.observe(n70, 1.0)
q0 = g.query(n2)
q1 = g.query(n16)
q2 = g.query(n21)
q3 = g.query(n26)
q4 = g.query(n31)
q5 = g.query(n36)
q6 = g.query(n41)
q7 = g.query(n46)
q8 = g.query(n51)
q9 = g.query(n56)
q10 = g.query(n61)
q11 = g.query(n66)
q12 = g.query(n6)
q13 = g.query(n7)
q14 = g.query(n10)
q15 = g.query(n11)
"""
        self.assertEqual(expected.strip(), observed.strip())
