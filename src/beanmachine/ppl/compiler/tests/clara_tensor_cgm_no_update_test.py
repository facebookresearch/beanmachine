# Copyright (c) Facebook, Inc. and its affiliates.
# """End-to-end test for a CLARA CGM model"""
import unittest

import beanmachine.ppl as bm
import torch.distributions as dist
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor


## New

# DATA
NUM_CLASS = 2  # num_classes (For Dirichlet may be need at least two)
# TODO: Clarify number of lablers is implicit
NUM_ITEMS = 1  # number of items
PREV_PRIOR = tensor([1.0, 0.0])  # prior on prevalence
# PREV_PRIOR is a list of length NUM_CLASSES
CONF_MATRIX_PRIOR = tensor([1.0, 0.0])  # prior on confusion matrix
# CONF_MATRIX_PRIOR is a list of length NUM_CLASS
# TODO: Does Dirichlet support 2d matrices?
# TODO: Is it really necessary to reject dirichlet on tensor([1])?
IDX_RATINGS = [[0]]  # indexed ratings that labelers assigned to items
IDX_LABELERS = [[0]]  # indexed list of labelers who labeled items
EXPERT_CONF_MATRIX = tensor(
    [[0.99, 0.01], [0.01, 0.99]]
)  # confusion matrix of an expert (if we have true ratings)
# EXPERT_CONF_MATRIX is of size NUM_CLASS x NUM_CLASS
# Row (first) index is true class, and column (second) index is observed
IDX_TRUE_RATINGS = [0]
# Of size NUM_ITEMS
# Represents true class of items by a perfect labler
# When information is missing, use value NUM_CLASS

# MODEL


@bm.random_variable
def prevalence():
    # Dirichlet distribution support is implemented in Beanstalk but not yet landed.
    return dist.Dirichlet(PREV_PRIOR)


@bm.random_variable
def confusion_matrix(labeler, true_class):
    return dist.Dirichlet(CONF_MATRIX_PRIOR)  # size: NUM_CLASSES


# log of the unnormalized item probs
# log P(true label of item i = k | labels)
# shape: [NUM_ITEMS, NUM_CLASSES]
@bm.functional
def log_item_prob(i, k):
    # Indexing into a simplex with a constant is implemented
    # but not yet landed
    prob = prevalence()[k].log()
    for r in range(len(IDX_RATINGS[i])):
        label = IDX_RATINGS[i][r]
        labeler = IDX_LABELERS[i][r]
        prob = prob + confusion_matrix(labeler, k)[label].log()
    if IDX_TRUE_RATINGS[i] != NUM_CLASS:  # Value NUM_CLASS means missing value
        prob = prob + EXPERT_CONF_MATRIX[k, IDX_TRUE_RATINGS[i]].log()
    return prob


# log of joint prob of labels, prev, conf_matrix
@bm.random_variable
def target():
    joint_log_prob = 0
    for i in range(NUM_ITEMS):
        # logsumexp on a newly-constructed tensor with stochastic
        # elements has limited support but this should work:
        log_probs = tensor(
            # TODO: Hard-coded k in {0,1}
            # [log_item_prob(i, 0), log_item_prob(i, 1), log_item_prob(i, 2)]
            # [log_item_prob(i, 0), log_item_prob(i, 1)]
            [log_item_prob(i, k) for k in range(NUM_CLASS)]
        )
        joint_log_prob = joint_log_prob + log_probs.logsumexp(0)
    return dist.Bernoulli(joint_log_prob.exp())


observations = {target(): tensor(1.0)}
queries = [
    log_item_prob(0, 0),  # Ideally, all the other elements too
    prevalence(),
    confusion_matrix(0, 0),  # Ideally, all the other elements too
]
ssrw = "SingleSiteRandomWalk"
bmgi = "BMG inference"
both = {ssrw, bmgi}
# TODO: Replace 4th param of expecteds by more methodical calculation
expecteds = [
    (prevalence(), both, 0.5000, 0.001),
    (confusion_matrix(0, 0), both, 0.5000, 0.001),
    (log_item_prob(0, 0), {ssrw}, -1.3863, 0.5),
    (log_item_prob(0, 0), {bmgi}, -1.0391, 0.5),
]


class ClaraTest(unittest.TestCase):
    def test_clara_tensor_cgm_no_update_inference(self) -> None:
        """Check BM and BMG inference both terminate"""

        self.maxDiff = None
        num_samples = 10

        # First, let's see how the model fairs with Random Walk inference
        inference = bm.SingleSiteRandomWalk()  # or NUTS
        mcsamples = inference.infer(queries, observations, num_samples)

        for rand_var, inferences, value, delta in expecteds:
            if ssrw in inferences:
                samples = mcsamples[rand_var]
                observed = samples.mean()
                expected = tensor([value])
                self.assertAlmostEqual(first=observed, second=expected, delta=delta)

        # Second, let's see how it fairs with the bmg inference
        inference = BMGInference()
        mcsamples = inference.infer(queries, observations, num_samples)

        for rand_var, inferences, value, delta in expecteds:
            if bmgi in inferences:
                samples = mcsamples[rand_var]
                observed = samples.mean()
                expected = tensor([value])
                self.assertAlmostEqual(first=observed, second=expected, delta=delta)

    def test_clara_tensor_cgm_no_update_to_dot_cpp_python(self) -> None:
        self.maxDiff = None
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N00[label="[1.0,0.0]"];
  N01[label=Dirichlet];
  N02[label=Sample];
  N03[label=Sample];
  N04[label=Sample];
  N05[label=0];
  N06[label=index];
  N07[label=Log];
  N08[label=index];
  N09[label=Log];
  N10[label=-0.010050326585769653];
  N11[label="+"];
  N12[label=1];
  N13[label=index];
  N14[label=Log];
  N15[label=index];
  N16[label=Log];
  N17[label=-4.605170249938965];
  N18[label="+"];
  N19[label=LogSumExp];
  N20[label=ToReal];
  N21[label=Exp];
  N22[label=ToProb];
  N23[label=Bernoulli];
  N24[label=Sample];
  N25[label="Observation True"];
  N26[label=Query];
  N27[label=Query];
  N28[label=Query];
  N00 -> N01;
  N01 -> N02;
  N01 -> N03;
  N01 -> N04;
  N02 -> N06;
  N02 -> N13;
  N02 -> N27;
  N03 -> N08;
  N03 -> N28;
  N04 -> N15;
  N05 -> N06;
  N05 -> N08;
  N05 -> N15;
  N06 -> N07;
  N07 -> N11;
  N08 -> N09;
  N09 -> N11;
  N10 -> N11;
  N11 -> N19;
  N11 -> N26;
  N12 -> N13;
  N13 -> N14;
  N14 -> N18;
  N15 -> N16;
  N16 -> N18;
  N17 -> N18;
  N18 -> N19;
  N19 -> N20;
  N20 -> N21;
  N21 -> N22;
  N22 -> N23;
  N23 -> N24;
  N24 -> N25;
}
        """
        self.assertEqual(observed.strip(), expected.strip())

        observed = BMGInference().to_cpp(queries, observations)
        expected = """
graph::Graph g;
Eigen::MatrixXd m0(2, 1)
m0 << 1.0, 0.0;
uint n0 = g.add_constant_pos_matrix(m0);
uint n1 = g.add_distribution(
  graph::DistributionType::DIRICHLET,
  graph::ValueType(
    graph::VariableType::COL_SIMPLEX_MATRIX,
    graph::AtomicType::PROBABILITY,
    2,
    1
  )
  std::vector<uint>({n0}));
uint n2 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n3 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n4 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n5 = g.add_constant(0);
uint n6 = g.add_operator(
  graph::OperatorType::INDEX, std::vector<uint>({n2, n5}));
uint n7 = g.add_operator(
  graph::OperatorType::LOG, std::vector<uint>({n6}));
uint n8 = g.add_operator(
  graph::OperatorType::INDEX, std::vector<uint>({n3, n5}));
uint n9 = g.add_operator(
  graph::OperatorType::LOG, std::vector<uint>({n8}));
uint n10 = g.add_constant_neg_real(-0.010050326585769653);
n11 = g.add_operator(
  graph::OperatorType::ADD,
  std::vector<uint>({n7, n9, n10}));
uint n12 = g.add_constant(1);
uint n13 = g.add_operator(
  graph::OperatorType::INDEX, std::vector<uint>({n2, n12}));
uint n14 = g.add_operator(
  graph::OperatorType::LOG, std::vector<uint>({n13}));
uint n15 = g.add_operator(
  graph::OperatorType::INDEX, std::vector<uint>({n4, n5}));
uint n16 = g.add_operator(
  graph::OperatorType::LOG, std::vector<uint>({n15}));
uint n17 = g.add_constant_neg_real(-4.605170249938965);
n18 = g.add_operator(
  graph::OperatorType::ADD,
  std::vector<uint>({n14, n16, n17}));
n19 = g.add_operator(
  graph::OperatorType::LOGSUMEXP,
  std::vector<uint>({n11, n18}));
uint n20 = g.add_operator(
  graph::OperatorType::TO_REAL, std::vector<uint>({n19}));
uint n21 = g.add_operator(
  graph::OperatorType::EXP, std::vector<uint>({n20}));
uint n22 = g.add_operator(
  graph::OperatorType::TO_PROBABILITY, std::vector<uint>({n21}));
uint n23 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n22}));
uint n24 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n23}));
g.observe([n24], true);
g.query(n11);
g.query(n2);
g.query(n3);
"""
        self.assertEqual(observed.strip(), expected.strip())

        observed = BMGInference().to_python(queries, observations)
        expected = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_pos_matrix(tensor([[1.0],[0.0]]))
n1 = g.add_distribution(
  graph.DistributionType.DIRICHLET,
  graph.ValueType(
    graph.VariableType.COL_SIMPLEX_MATRIX,
    graph.AtomicType.PROBABILITY,
    2,
    1,
  ),
  [n0],
)
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n3 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n4 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n5 = g.add_constant(0)
n6 = g.add_operator(graph.OperatorType.INDEX, [n2, n5])
n7 = g.add_operator(graph.OperatorType.LOG, [n6])
n8 = g.add_operator(graph.OperatorType.INDEX, [n3, n5])
n9 = g.add_operator(graph.OperatorType.LOG, [n8])
n10 = g.add_constant_neg_real(-0.010050326585769653)
n11 = g.add_operator(
  graph.OperatorType.ADD,
  [n7, n9, n10])
n12 = g.add_constant(1)
n13 = g.add_operator(graph.OperatorType.INDEX, [n2, n12])
n14 = g.add_operator(graph.OperatorType.LOG, [n13])
n15 = g.add_operator(graph.OperatorType.INDEX, [n4, n5])
n16 = g.add_operator(graph.OperatorType.LOG, [n15])
n17 = g.add_constant_neg_real(-4.605170249938965)
n18 = g.add_operator(
  graph.OperatorType.ADD,
  [n14, n16, n17])
n19 = g.add_operator(
  graph.OperatorType.LOGSUMEXP,
  [n11, n18])
n20 = g.add_operator(graph.OperatorType.TO_REAL, [n19])
n21 = g.add_operator(graph.OperatorType.EXP, [n20])
n22 = g.add_operator(graph.OperatorType.TO_PROBABILITY, [n21])
n23 = g.add_distribution(
  graph.DistributionType.BERNOULLI,
  graph.AtomicType.BOOLEAN,
  [n22])
n24 = g.add_operator(graph.OperatorType.SAMPLE, [n23])
g.observe(n24, True)
g.query(n11)
g.query(n2)
g.query(n3)
"""
        self.assertEqual(observed.strip(), expected.strip())
