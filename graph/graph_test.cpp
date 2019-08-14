// Copyright (c) Facebook, Inc. and its affiliates.
#include <array>
#include <tuple>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <beanmachine/graph/graph.h>

using namespace ::testing;

using namespace beanmachine;

TEST(testgraph, support) {
  graph::Graph g;
  // c1 and c2 are parents of o1 -> d1 -> o2
  uint c1 = g.add_constant(.3);
  uint c2 = g.add_constant(.4);
  uint o1 = g.add_operator(
      graph::OperatorType::MULTIPLY, std::vector<uint>({c1, c2}));
  uint d1 = g.add_distribution(
      graph::DistributionType::BERNOULLI,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>{o1});
  uint o2 =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d1}));
  // c3 and c4 are parents of o3 -> d2 -> o4
  uint c3 = g.add_constant(.3);
  uint c4 = g.add_constant(.4);
  uint o3 = g.add_operator(
      graph::OperatorType::MULTIPLY, std::vector<uint>({c3, c4}));
  uint d2 = g.add_distribution(
      graph::DistributionType::BERNOULLI,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>{o3});
  uint o4 =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d2}));
  // o2 and o4 -> o5
  uint o5 = g.add_operator(
      graph::OperatorType::MULTIPLY, std::vector<uint>({o2, o4}));
  uint d3 = g.add_distribution(
      graph::DistributionType::BERNOULLI,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({o5}));
  uint o6 =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d3}));
  uint o7 =
      g.add_operator(graph::OperatorType::TO_REAL, std::vector<uint>({o6}));
  uint o8 = g.add_operator(
      graph::OperatorType::MULTIPLY, std::vector<uint>({c3, o7}));
  // observe o4 so support includes c3, c4, o3, d2, o4
  g.observe(o4, true);
  auto supp = g.compute_support();
  EXPECT_EQ(supp.size(), 5);
  EXPECT_EQ(*supp.begin(), c3);
  EXPECT_EQ(*supp.rbegin(), o4);
  EXPECT_TRUE(supp.find(o3) != supp.end());
  std::list<uint> det_nodes;
  std::list<uint> sto_nodes;
  std::tie(det_nodes, sto_nodes) = g.compute_descendants(c3);
  // c3 -> det: c3, o3, d2, o8 sto: o4
  EXPECT_EQ(det_nodes.size(), 4);
  EXPECT_EQ(det_nodes.front(), c3);
  EXPECT_EQ(det_nodes.back(), o8);
  EXPECT_EQ(sto_nodes.size(), 1);
  EXPECT_EQ(sto_nodes.front(), o4);
  // query o5 so support is now 11 nodes:
  //   c1, c2, o1, d1, o2, c3, c4, o3, d2, o4, o5
  g.query(o5);
  supp = g.compute_support();
  EXPECT_EQ(supp.size(), 11);
  // o4 -> det: o5, d3 sto: o4, o6
  // note: o7 and o8 are not in the descendants of o4
  // because the descendant chain gets cut off at the stochastic node o6
  std::tie(det_nodes, sto_nodes) = g.compute_descendants(o4);
  EXPECT_EQ(det_nodes.size(), 2);
  EXPECT_EQ(det_nodes.front(), o5);
  EXPECT_EQ(det_nodes.back(), d3);
  EXPECT_EQ(sto_nodes.size(), 2);
  EXPECT_EQ(sto_nodes.front(), o4);
  EXPECT_EQ(sto_nodes.back(), o6);
}

TEST(testgraph, infer_arithmetic) {
  graph::Graph g;
  uint c1 = g.add_constant(0.1);
  uint d1 = g.add_distribution(
      graph::DistributionType::BERNOULLI,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({c1}));
  uint o1 =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d1}));
  uint o2 =
      g.add_operator(graph::OperatorType::TO_REAL, std::vector<uint>({o1}));
  uint c2 = g.add_constant(0.8);
  uint o3 = g.add_operator(
      graph::OperatorType::MULTIPLY, std::vector<uint>({c2, o2}));
  uint c3 = g.add_constant(0.1);
  uint o4 =
      g.add_operator(graph::OperatorType::ADD, std::vector<uint>({c3, o3}));
  uint d2 = g.add_distribution(
      graph::DistributionType::BERNOULLI,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({o4}));
  uint o5 =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d2}));
  // P(o5|o1) = .9 when o1 is true and .1 when o1 is false
  g.observe(o5, true);
  g.query(o1);
  std::vector<std::vector<graph::AtomicValue>> samples =
      g.infer(100, graph::InferenceType::GIBBS);
  uint sum = 0;
  for (const auto& sample : samples) {
    const auto& s = sample.front();
    ASSERT_EQ(s.type, graph::AtomicType::BOOLEAN);
    sum += s._bool ? 1 : 0;
  }
  // less than 1 in million odds of failing these tests with correct infer
  EXPECT_LT(sum, 75);
  EXPECT_GT(sum, 25);
  std::vector<std::vector<graph::AtomicValue>> samples2 =
      g.infer(100, graph::InferenceType::REJECTION);
  sum = 0;
  for (const auto& sample : samples2) {
    const auto& s = sample.front();
    ASSERT_EQ(s.type, graph::AtomicType::BOOLEAN);
    sum += s._bool ? 1 : 0;
  }
  EXPECT_LT(sum, 75);
  EXPECT_GT(sum, 25);
}

TEST(testgraph, infer_bn) {
  graph::Graph g;
  // classic sprinkler BN, see the diagram here:
  // https://upload.wikimedia.org/wikipedia/commons/0/0e/SimpleBayesNet.svg
  uint c1 = g.add_constant(torch::from_blob((float[]){.8, .2}, {2}));
  uint d1 = g.add_distribution(
      graph::DistributionType::TABULAR,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({c1}));
  uint RAIN =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d1}));
  uint c2 =
      g.add_constant(torch::from_blob((float[]){.6, .4, .99, .01}, {2, 2}));
  uint d2 = g.add_distribution(
      graph::DistributionType::TABULAR,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({c2, RAIN}));
  uint SPRINKLER =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d2}));
  uint c3 = g.add_constant(
      torch::from_blob((float[]){1, 0, .2, .8, .1, .9, .01, .99}, {2, 2, 2}));
  uint d3 = g.add_distribution(
      graph::DistributionType::TABULAR,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({c3, SPRINKLER, RAIN}));
  uint GRASSWET =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d3}));
  g.observe(GRASSWET, true);
  g.query(RAIN);
  const std::vector<std::vector<graph::AtomicValue>>& samples =
      g.infer(100, graph::InferenceType::REJECTION);
  ASSERT_EQ(samples.size(), 100);
  uint sum = 0;
  for (const auto& sample : samples) {
    const auto& s = sample.front();
    sum += s._bool ? 1 : 0;
  }
  // true probability is approx .3577
  // less than 1 in million odds of failing these tests with correct infer
  EXPECT_LT(sum, 60);
  EXPECT_GT(sum, 10);
}
