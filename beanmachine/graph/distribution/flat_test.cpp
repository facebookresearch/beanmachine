// Copyright (c) Facebook, Inc. and its affiliates.
#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"


using namespace beanmachine::graph;


TEST(testdistrib, flat) {
  Graph g;
  auto real1 = g.add_constant(3.0);
  // negative test: flat has no parent
  EXPECT_THROW(
      g.add_distribution(DistributionType::FLAT, AtomicType::REAL, std::vector<uint>{real1}),
      std::invalid_argument);
  auto bool_dist = g.add_distribution(
      DistributionType::FLAT, AtomicType::BOOLEAN, std::vector<uint>{});
  auto bool_val = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{bool_dist});
  g.query(bool_val);
  auto prob_dist = g.add_distribution(
      DistributionType::FLAT, AtomicType::PROBABILITY, std::vector<uint>{});
  auto prob_val = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{prob_dist});
  g.query(prob_val);
  auto real_dist = g.add_distribution(
      DistributionType::FLAT, AtomicType::REAL, std::vector<uint>{});
  auto real_val = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{real_dist});
  g.query(real_val);
  auto pos_dist = g.add_distribution(
      DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{});
  auto pos_val = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{pos_dist});
  g.query(pos_val);
  auto natural_dist = g.add_distribution(
      DistributionType::FLAT, AtomicType::NATURAL, std::vector<uint>{});
  auto natural_val = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{natural_dist});
  g.query(natural_val);
  // negative test: Tensors are not supported
  EXPECT_THROW(
      g.add_distribution(
      DistributionType::FLAT, AtomicType::TENSOR, std::vector<uint>{}),
      std::invalid_argument);
  const std::vector<double>& means = g.infer_mean(100000, InferenceType::REJECTION);
  EXPECT_NEAR(means[0], 0.5, 0.01); // boolean
  EXPECT_NEAR(means[1], 0.5, 0.01); // probability
  // note: we don't have a test for real since this could be huge
  EXPECT_GT(means[3], 1000); // pos_real should be a big number!
  EXPECT_GT(means[4], 1000); // natural should be a big number!
  // log_probs and gradients are all zero for Flat distribution
  EXPECT_EQ(g.log_prob(bool_val), 0);
  EXPECT_EQ(g.log_prob(prob_val), 0);
  EXPECT_EQ(g.log_prob(real_val), 0);
  EXPECT_EQ(g.log_prob(pos_val), 0);
  EXPECT_EQ(g.log_prob(natural_val), 0);
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(bool_val, grad1, grad2);
  g.gradient_log_prob(prob_val, grad1, grad2);
  g.gradient_log_prob(real_val, grad1, grad2);
  g.gradient_log_prob(pos_val, grad1, grad2);
  g.gradient_log_prob(natural_val, grad1, grad2);
  EXPECT_EQ(grad1, 0);
  EXPECT_EQ(grad2, 0);
}
