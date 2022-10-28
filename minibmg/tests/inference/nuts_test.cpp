/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <memory>
#include "beanmachine/graph/global/global_mh.h"
#include "beanmachine/graph/global/nuts.h"
#include "beanmachine/minibmg/fluid_factory.h"
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/inference/global_state.h"

using namespace ::testing;
using namespace beanmachine::minibmg;
using beanmachine::graph::GlobalState;
using beanmachine::graph::NodeValue;
using beanmachine::graph::NUTS;

template <typename T>
requires Number<T> T expit(const T& x) {
  return 1 / (1 + exp(-x));
}

const int num_heads = 15;
const int num_tails = 1;
const int num_samples = 1000;
const int skip_samples = std::min(num_samples / 2, 500);
const int seed = 12345;

// Take a familiar-looking model and runs NUTS.
TEST(nuts_test, coin_flipping) {
  Graph::FluidFactory f;

  // We would like to use
  //
  //     auto d = beta(1, 1);
  //     auto s = sample(d);
  //
  // but we don't have transformations working quite right yet, which we would
  // need to use beta.  So we use a distribution that doesn't require it.

  auto s = expit(sample(normal(0, 100)));

  auto bn = bernoulli(s);
  for (int i = 0; i < num_heads; i++) {
    f.observe(sample(bn), 1);
  }
  for (int i = 0; i < num_tails; i++) {
    f.observe(sample(bn), 0);
  }
  f.query(s);
  auto graph = f.build();

  auto state = std::make_unique<MinibmgGlobalState>(graph);
  auto nuts = NUTS(std::move(state));

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<NodeValue>> infer_results =
      nuts.infer(/* num_samples = */ num_samples, /* seed = */ seed);
  auto finish = std::chrono::high_resolution_clock::now();
  auto time_in_microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(finish - start)
          .count();
  std::cout << fmt::format(
                   "minibmg NUTS: ran in {} s", time_in_microseconds / 1E6)
            << std::endl;

  // check that the results are as expected
  ASSERT_EQ(infer_results.size(), num_samples);
  double sum = 0;
  int count = 0;
  for (int i = skip_samples; i < num_samples; i++) {
    auto estimate = infer_results[i][0]._double;
    sum += estimate;
    count++;
  }

  double average = sum / num_samples;
  // the following is the actual computed value using bmg
  double expected =
      0.46432190477921237; // num_heads / (num_heads + num_tails + 0.0);
  ASSERT_NEAR(average, expected, 0.001);
}

// Take a familiar-looking model and runs NUTS using bmg.
TEST(nuts_test, coin_flipping_bmg) {
  using namespace beanmachine::graph;
  using Graph = beanmachine::graph::Graph;

  Graph g;

  //   auto s = expit(sample(normal(0, 100)));
  auto k0 = g.add_constant(0.0);
  auto k100 = g.add_constant_pos_real(100.0);
  auto normal = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {k0, k100});
  auto sample_normal = g.add_operator(OperatorType::SAMPLE, {normal});
  // s = (expit =) 1 / (1 + exp(-sample_normal))
  auto neg_sample = g.add_operator(OperatorType::NEGATE, {sample_normal});
  auto exp = g.add_operator(OperatorType::EXP, {neg_sample});
  auto k1 = g.add_constant_pos_real(1.0);
  auto denom = g.add_operator(OperatorType::ADD, {k1, exp});
  // At this point we would like to compute
  //    s = 1 / denom
  // but BMG has no divide or reciprocal operation.  So instead we compute
  //    s = exp(-log(denom))
  auto log_denom = g.add_operator(OperatorType::LOG, {denom});
  auto nld = g.add_operator(OperatorType::NEGATE, {log_denom});
  auto s0 = g.add_operator(OperatorType::EXP, {nld});
  auto s = g.add_operator(OperatorType::TO_PROBABILITY, {s0});

  auto bn =
      g.add_distribution(DistributionType::BERNOULLI, AtomicType::BOOLEAN, {s});

  for (int i = 0; i < num_heads; i++) {
    auto sample = g.add_operator(OperatorType::SAMPLE, {bn});
    g.observe(sample, true);
  }
  for (int i = 0; i < num_tails; i++) {
    auto sample = g.add_operator(OperatorType::SAMPLE, {bn});
    g.observe(sample, false);
  }

  g.query(s);
  auto nuts = NUTS(g);

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<NodeValue>> infer_results =
      nuts.infer(/* num_samples = */ num_samples, /* seed = */ seed);
  auto finish = std::chrono::high_resolution_clock::now();
  auto time_in_microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(finish - start)
          .count();
  std::cout << fmt::format(
                   "    bmg NUTS: ran in {} s", time_in_microseconds / 1E6)
            << std::endl;

  // check that the results are as expected
  ASSERT_EQ(infer_results.size(), num_samples);
  double sum = 0;
  int count = 0;
  for (int i = skip_samples; i < num_samples; i++) {
    auto estimate = infer_results[i][0]._double;
    sum += estimate;
    count++;
  }

  double average = sum / num_samples;
  // the following is the actual computed value using bmg
  double expected =
      0.46432190477921237; // num_heads / (num_heads + num_tails + 0.0);
  ASSERT_NEAR(average, expected, 0.001);
}
