// Copyright (c) Facebook, Inc. and its affiliates.
#include <array>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "beanmachine/graph/bernoulli.h"
#include "beanmachine/graph/bernoulli_noisy_or.h"
#include "beanmachine/graph/beta.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/tabular.h"


using namespace beanmachine;

#define LOG_ZERO_PT_9 ((double)-0.10536051565782628)
#define LOG_ZERO_PT_1 ((double)-2.3025850929940455)

TEST(testdistrib, bernoulli) {
  auto p1 = graph::AtomicValue(0.1);
  graph::ConstNode cnode1(p1);
  distribution::Bernoulli dnode1(
      graph::AtomicType::BOOLEAN, std::vector<graph::Node*>{&cnode1});
  dnode1.in_nodes.push_back(&cnode1);
  auto zero = graph::AtomicValue(false);
  auto one = graph::AtomicValue(true);

  EXPECT_NEAR(LOG_ZERO_PT_9, dnode1.log_prob(zero), 1e-3);
  EXPECT_NEAR(LOG_ZERO_PT_1, dnode1.log_prob(one), 1e-3);
}

TEST(testdistrib, bernoulli_noisy_or) {
  // Define log1mexp(x) = log(1 - exp(-x))
  // then log1mexp(1e-20) = -46.051701859880914
  // and log1mexp(40) = -4.248354255291589e-18
  // We will use the above facts in this test

  // first distribution
  auto p1 = graph::AtomicValue(1e-20);
  graph::ConstNode cnode1(p1);
  distribution::BernoulliNoisyOr dnode1(
      graph::AtomicType::BOOLEAN, std::vector<graph::Node*>{&cnode1});
  dnode1.in_nodes.push_back(&cnode1);
  auto zero = graph::AtomicValue(false);
  auto one = graph::AtomicValue(true);

  EXPECT_EQ(-1e-20, dnode1.log_prob(zero));
  EXPECT_NEAR(-46.05, dnode1.log_prob(one), 0.01);

  // second disgribution
  auto p2 = graph::AtomicValue(40.0);
  graph::ConstNode cnode2(p2);
  distribution::BernoulliNoisyOr dnode2(
      graph::AtomicType::BOOLEAN, std::vector<graph::Node*>{&cnode1});
  dnode2.in_nodes.push_back(&cnode2);

  EXPECT_EQ(-40, dnode2.log_prob(zero));
  EXPECT_NEAR(-4.248e-18, dnode2.log_prob(one), 0.001e-18);
}

TEST(testdistrib, tabular) {
  // clang-format off
  std::array<float, 4> mat_arr = {
      0.9, 0.1,
      0.1, 0.9};
  // clang-format on
  torch::Tensor mat = torch::from_blob(mat_arr.data(), {2, 2});
  graph::ConstNode cnode1(graph::AtomicValue{mat});
  graph::ConstNode cnode2(graph::AtomicValue{true});
  distribution::Tabular dnode1(
      graph::AtomicType::BOOLEAN, std::vector<graph::Node*>{&cnode1, &cnode2});
  dnode1.in_nodes.push_back(&cnode1);
  dnode1.in_nodes.push_back(&cnode2);
  auto zero = graph::AtomicValue(false);
  auto one = graph::AtomicValue(true);

  EXPECT_NEAR(LOG_ZERO_PT_1, dnode1.log_prob(zero), 1e-3);
  EXPECT_NEAR(LOG_ZERO_PT_9, dnode1.log_prob(one), 1e-3);

  cnode2.value = graph::AtomicValue(false);
  EXPECT_NEAR(LOG_ZERO_PT_9, dnode1.log_prob(zero), 1e-3);
  EXPECT_NEAR(LOG_ZERO_PT_1, dnode1.log_prob(one), 1e-3);
}

TEST(testdistrib, beta) {
  auto a = graph::AtomicValue(1.1);
  auto b = graph::AtomicValue(5.0);
  graph::ConstNode cnode_a(a);
  graph::ConstNode cnode_b(b);
  distribution::Beta dnode1(
      graph::AtomicType::PROBABILITY,
      std::vector<graph::Node*>{&cnode_a, &cnode_b});
  dnode1.in_nodes.push_back(&cnode_a);
  dnode1.in_nodes.push_back(&cnode_b);
  auto prob = graph::AtomicValue(graph::AtomicType::PROBABILITY, 0.2);
  // This value of 0.7773 was checked from PyTorch
  EXPECT_NEAR(0.7773, dnode1.log_prob(prob), 1e-3);
}
