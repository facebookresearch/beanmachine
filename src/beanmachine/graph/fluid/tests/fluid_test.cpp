/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <iostream>

#include "beanmachine/graph/fluid/fluid.h"
#include "beanmachine/graph/util.h"

using namespace ::testing;

using namespace beanmachine::graph;
using namespace beanmachine::util;
using namespace beanmachine::graph::fluid;

TEST(fluid_test, basic_test) {
  auto p = sample(beta(2, 2));
  auto toss1 = sample(bernoulli(p));
  auto toss2 = sample(bernoulli(p));

  Graph graph;
  observe(toss1, true, graph);
  observe(toss2, true, graph);
  query(p, graph);

  auto means = graph.infer_mean(5000, InferenceType::NMC, 123);
  std::cout << join(means) << std::endl;
  ASSERT_NEAR(means[0], 0.66, 1e-2);
}

TEST(fluid_test, unused_values_test) {
  // in this test, the node in 'not_used' is not included
  // in the graph and must be deallocated by the pool mechanism.
  // To make it a bit more interesting, we make it depend on
  // a node ('p_prior') that does get moved into the Graph
  // and does not need to be deallocated by the pool mechanism.
  auto p_prior = beta(2, 2);
  auto not_used = sample(p_prior);
  auto p = sample(p_prior);
  auto toss1 = sample(bernoulli(p));
  auto toss2 = sample(bernoulli(p));

  Graph graph;
  observe(toss1, true, graph);
  observe(toss2, true, graph);
  query(p, graph);

  auto means = graph.infer_mean(5000, InferenceType::NMC, 123);
  std::cout << join(means) << std::endl;
  ASSERT_NEAR(means[0], 0.66, 1e-2);
}

TEST(fluid_test, other_node_types_test) {
  auto mu = sample(normal(5.0, 2.0));
  auto px = normal(mu, 0.1);
  auto x1 = sample(px);
  auto x2 = sample(px);
  auto y = -(-(2 * x1 - x2));
  auto z = sample(normal(y, 0.1));

  Graph graph;
  observe(z, 10.0, graph);
  query(mu, graph);

  auto means = graph.infer_mean(5000, InferenceType::NMC, 123);
  std::cout << join(means) << std::endl;
  ASSERT_NEAR(means[0], 9.86, 1e-2);
}
