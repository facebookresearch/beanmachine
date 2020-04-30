// Copyright (c) Facebook, Inc. and its affiliates.
#include <array>
#include <tuple>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "beanmachine/graph/graph.h"

using namespace beanmachine;

TEST(testgraph, infer_arithmetic) {
  graph::Graph g;
  uint c1 = g.add_constant_probability(0.1);
  uint d1 = g.add_distribution(
      graph::DistributionType::BERNOULLI,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({c1}));
  uint o1 =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d1}));
  uint o2 =
      g.add_operator(graph::OperatorType::TO_POS_REAL, std::vector<uint>({o1}));
  uint c2 = g.add_constant_pos_real(0.8);
  uint o3 = g.add_operator(
      graph::OperatorType::MULTIPLY, std::vector<uint>({c2, o2}));
  uint c3 = g.add_constant_pos_real(0.1);
  uint o4 =
      g.add_operator(graph::OperatorType::ADD, std::vector<uint>({c3, o3}));
  uint d2 = g.add_distribution(
      graph::DistributionType::BERNOULLI_NOISY_OR,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({o4}));
  uint o5 =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d2}));
  // P(o5|o1=T) = 1 - exp(-.9)=0.5934 and P(o5|o1=F) = 1-exp(-.1)=0.09516
  // Since P(o1=T)=0.1 and P(o1=F)=0.9. Therefore P(o5=T,o1=T) = 0.05934,
  // P(o5=T,o1=F) = 0.08564 and P(o1=T | o5=T) = 0.4093
  g.observe(o5, true);
  g.query(o1);
  std::vector<std::vector<graph::AtomicValue>> samples =
      g.infer(100, graph::InferenceType::GIBBS);
  int sum = 0;
  for (const auto& sample : samples) {
    const auto& s = sample.front();
    ASSERT_EQ(s.type, graph::AtomicType::BOOLEAN);
    sum += s._bool ? 1 : 0;
  }
  // less than 1 in million odds of failing these tests with correct infer
  EXPECT_LT(sum, 65);
  EXPECT_GT(sum, 15);
  // infer_mean should give almost exactly the same answer
  std::vector<double> means = g.infer_mean(100, graph::InferenceType::GIBBS);
  EXPECT_TRUE(std::abs(sum - int(means[0] * 100)) <= 1);
  // repeat the test with rejection sampling
  std::vector<std::vector<graph::AtomicValue>> samples2 =
      g.infer(100, graph::InferenceType::REJECTION);
  sum = 0;
  for (const auto& sample : samples2) {
    const auto& s = sample.front();
    ASSERT_EQ(s.type, graph::AtomicType::BOOLEAN);
    sum += s._bool ? 1 : 0;
  }
  EXPECT_LT(sum, 65);
  EXPECT_GT(sum, 15);
  // infer_mean should give the same answer
  std::vector<double> means2 =
      g.infer_mean(100, graph::InferenceType::REJECTION);
  EXPECT_TRUE(std::abs(sum - int(means2[0] * 100)) <= 1);
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
