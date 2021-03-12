# Copyright (c) Facebook, Inc. and its affiliates.
# """End-to-end test for a CLARA CGM model"""
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli, Beta

## New

# DATA
K = 1  # num_classes
NUM_ITEMS = 1  # number of items
PREV_PRIOR = 1  # prior on prevalence
# PREV_PRIOR is a list of length K
CONF_MATRIX_PRIOR = 1  # prior on confusion matrix
# CONF_MATRIX_PRIOR is a list of length K
IDX_RATINGS = 1  # indexed ratings that labelers assigned to items
IDX_LABELERS = 1  # indexed list of labelers who labeled items
EXPERT_CONF_MATRIX = 1  # confusion matrix of an expert (if we have true ratings)
IDX_TRUE_RATINGS = 1  # indexed true ratings (non-zero if we have true ratings)

# MODEL


@bm.random_variable
def prevalence():
    # Dirichlet distribution support is implemented in Beanstalk but not yet landed.
    return dist.Dirichlet(PREV_PRIOR)


@bm.random_variable
def confusion_matrix(labeler, true_class):
    return dist.Dirichlet(CONF_MATRIX_PRIOR)  # size: K


# log of the unnormalized item probs
# log P(true label of item i = k | labels)
# shape: [NUM_ITEMS, K]
@bm.functional
def log_item_prob():
    # Indexing into a simplex with a constant is implemented
    # but not yet landed.
    item_probs = torch.zeros((NUM_ITEMS, K))
    for i in range(NUM_ITEMS):
        for k in range(K):
            prob = torch.log(prevalence()[k])
            for r in range(len(IDX_RATINGS[i])):
                label = IDX_RATINGS[i][r]
                labeler = IDX_LABELERS[i][r]
                prob = prob + torch.log(confusion_matrix(labeler, k)[label])
    if IDX_TRUE_RATINGS[i] != 0:
        prob = prob + torch.log(EXPERT_CONF_MATRIX[k, IDX_TRUE_RATINGS[i]])
        # This operation is not implemented in BMG or in the compiler.
        # That is, gradual construction of a 2-d tensor whose elements
        # are all graph nodes.
        item_probs[i, k] = prob
    return item_probs


# log of joint prob of labels, prev, conf_matrix
@bm.random_variable
def target():
    joint_log_prob = 0
    for i in range(NUM_ITEMS):
        # This operation is not implemented in BMG or in the compiler.
        # That is, gradual construction of a 2-d tensor whose elements
        # are all graph nodes followed by extraction of a row from that
        # tensor.
        joint_log_prob = joint_log_prob + torch.logsumexp(log_item_prob()[i])
    return dist.Bernoulli(torch.exp(joint_log_prob))


observations = {target(): tensor(1.0)}
queries = [log_item_prob(), prevalence()] + ["...for loop for all confusion matrix..."]

# OLD


@bm.random_variable
def beta():
    return Beta(2.0, 2.0)


@bm.random_variable
def flip(n):
    return Bernoulli(beta())


class CoinFlipTest(unittest.TestCase):
    def test_clara_tensor_cgm_coin_flip_inference(self) -> None:
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

    def test_clara_tensor_cgm_coin_flip_to_dot_cpp_python(self) -> None:
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
        self.assertEqual(observed.strip(), expected.strip())

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
uint n6 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n3}));
g.observe([n6], false);
uint n8 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n3}));
g.observe([n8], true);
uint n10 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n3}));
g.observe([n10], false);
g.query(n2);"""
        self.assertEqual(observed.strip(), expected.strip())

        observed = BMGInference().to_python(queries, observations)
        expected = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_pos_real(2.0)
n1 = g.add_distribution(
  graph.DistributionType.BETA,
  graph.AtomicType.PROBABILITY,
  [n0, n0])
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n3 = g.add_distribution(
  graph.DistributionType.BERNOULLI,
  graph.AtomicType.BOOLEAN,
  [n2])
n4 = g.add_operator(graph.OperatorType.SAMPLE, [n3])
g.observe(n4, False)
n6 = g.add_operator(graph.OperatorType.SAMPLE, [n3])
g.observe(n6, False)
n8 = g.add_operator(graph.OperatorType.SAMPLE, [n3])
g.observe(n8, True)
n10 = g.add_operator(graph.OperatorType.SAMPLE, [n3])
g.observe(n10, False)
g.query(n2)
        """
        self.assertEqual(observed.strip(), expected.strip())
