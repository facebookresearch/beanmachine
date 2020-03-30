// Copyright (c) Facebook, Inc. and its affiliates.
#include <array>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "beanmachine/graph/bernoulli.h"
#include "beanmachine/graph/bernoulli_noisy_or.h"
#include "beanmachine/graph/beta.h"
#include "beanmachine/graph/binomial.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/tabular.h"


using namespace beanmachine;

#define LOG_ZERO_PT_9 ((double)-0.10536051565782628)
#define LOG_ZERO_PT_1 ((double)-2.3025850929940455)

TEST(testdistrib, bernoulli) {
  auto p1 = graph::AtomicValue(graph::AtomicType::PROBABILITY, 0.1);
  graph::ConstNode cnode1(p1);
  // positive test
  distribution::Bernoulli dnode1(
      graph::AtomicType::BOOLEAN, std::vector<graph::Node*>{&cnode1});
  dnode1.in_nodes.push_back(&cnode1);
  auto zero = graph::AtomicValue(false);
  auto one = graph::AtomicValue(true);
  EXPECT_NEAR(LOG_ZERO_PT_9, dnode1.log_prob(zero), 1e-3);
  EXPECT_NEAR(LOG_ZERO_PT_1, dnode1.log_prob(one), 1e-3);
  // negative test for return type
  EXPECT_THROW(distribution::Bernoulli(
        graph::AtomicType::REAL,
        std::vector<graph::Node*>{&cnode1}),
      std::invalid_argument);
  // negative tests for number of arguments
  EXPECT_THROW(distribution::Bernoulli(
        graph::AtomicType::BOOLEAN,
        std::vector<graph::Node*>{}),
      std::invalid_argument);
  EXPECT_THROW(distribution::Bernoulli(
        graph::AtomicType::BOOLEAN,
        std::vector<graph::Node*>{&cnode1, &cnode1}),
      std::invalid_argument);
  // negative test on datatype of parents
  auto p2 = graph::AtomicValue(graph::AtomicType::POS_REAL, 0.1);
  graph::ConstNode cnode2(p2);
  EXPECT_THROW(distribution::Bernoulli(
        graph::AtomicType::BOOLEAN,
        std::vector<graph::Node*>{&cnode2}),
      std::invalid_argument);
}

TEST(testdistrib, bernoulli_noisy_or) {
  // Define log1mexp(x) = log(1 - exp(-x))
  // then log1mexp(1e-10) = -23.02585084720009
  // and log1mexp(40) = -4.248354255291589e-18
  // We will use the above facts in this test

  // first distribution
  const double small_value = 1e-10;
  auto p1 = graph::AtomicValue(graph::AtomicType::POS_REAL, small_value);
  graph::ConstNode cnode1(p1);
  distribution::BernoulliNoisyOr dnode1(
      graph::AtomicType::BOOLEAN, std::vector<graph::Node*>{&cnode1});
  dnode1.in_nodes.push_back(&cnode1);
  auto zero = graph::AtomicValue(false);
  auto one = graph::AtomicValue(true);

  EXPECT_EQ(-small_value, dnode1.log_prob(zero));
  EXPECT_NEAR(-23.02, dnode1.log_prob(one), 0.01);

  // second distribution
  const double large_value = 40.0;
  auto p2 = graph::AtomicValue(graph::AtomicType::POS_REAL, large_value);
  graph::ConstNode cnode2(p2);
  distribution::BernoulliNoisyOr dnode2(
      graph::AtomicType::BOOLEAN, std::vector<graph::Node*>{&cnode1});
  dnode2.in_nodes.push_back(&cnode2);

  EXPECT_EQ(-large_value, dnode2.log_prob(zero));
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
  auto a = graph::AtomicValue(graph::AtomicType::POS_REAL, 1.1);
  auto b = graph::AtomicValue(graph::AtomicType::POS_REAL, 5.0);
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

TEST(testdistrib, binomial) {
  auto n = graph::AtomicValue((graph::natural_t) 10);
  auto p = graph::AtomicValue(graph::AtomicType::PROBABILITY, 0.5);
  graph::ConstNode cnode_n(n);
  graph::ConstNode cnode_p(p);
  distribution::Binomial dnode1(
      graph::AtomicType::NATURAL,
      std::vector<graph::Node*>{&cnode_n, &cnode_p});
  dnode1.in_nodes.push_back(&cnode_n);
  dnode1.in_nodes.push_back(&cnode_p);
  auto k0 = graph::AtomicValue((graph::natural_t) 0);
  auto k5 = graph::AtomicValue((graph::natural_t) 5);
  auto k11 = graph::AtomicValue((graph::natural_t) 11);
  EXPECT_TRUE(!std::isfinite(dnode1.log_prob(k11)));
  EXPECT_NEAR(10 * log(0.5), dnode1.log_prob(k0), 1e-2);
  // This value of -1.4020 was checked from PyTorch
  EXPECT_NEAR(-1.4020, dnode1.log_prob(k5), 1e-2);
  // negative test for return type of Binomial
  EXPECT_THROW(distribution::Binomial(
        graph::AtomicType::REAL,
        std::vector<graph::Node*>{&cnode_n, &cnode_p}),
      std::invalid_argument);
  // negative tests for number of arguments
  EXPECT_THROW(distribution::Binomial(
        graph::AtomicType::NATURAL,
        std::vector<graph::Node*>{&cnode_n}),
      std::invalid_argument);
  // negative test on data type of parents
  auto p2 = graph::AtomicValue(graph::AtomicType::REAL, 0.5);
  graph::ConstNode cnode_p2(p2);
  EXPECT_THROW(distribution::Binomial(
        graph::AtomicType::NATURAL,
        std::vector<graph::Node*>{&cnode_n, &cnode_p2}),
      std::invalid_argument);
}
