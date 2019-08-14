// Copyright (c) Facebook, Inc. and its affiliates.
#include <array>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <beanmachine/graph/bernoulli.h>
#include <beanmachine/graph/graph.h>
#include <beanmachine/graph/tabular.h>

using namespace ::testing;

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
