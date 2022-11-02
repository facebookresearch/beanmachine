/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <ctime>
#include <iostream>
#include <stdexcept>

#include "beanmachine/graph/distribution/bernoulli.h"
#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/marginalization/marginalization_extensional.h"
#include "beanmachine/graph/operator/operator.h"
#include "beanmachine/graph/util.h"

using namespace ::testing;

using namespace beanmachine;
using namespace graph;
using namespace std;
using namespace util;

std::tuple<NodeID, NodeID> make_childless_base_graph(Graph& graph) {
  // discrete = sample(Bernoulli(0.4))
  // real_discrete = to_real(discrete)
  // Normal(real_discrete, 0.5))
  auto point_four = graph.add_constant_probability(0.4);
  auto stddev = graph.add_constant_pos_real(1.0);
  auto bernoulli = graph.add_distribution(
      DistributionType::BERNOULLI, AtomicType::BOOLEAN, {point_four});
  auto discrete =
      graph.add_operator(OperatorType::SAMPLE, {bernoulli}); // node_id = 3
  auto real_discrete = graph.add_operator(OperatorType::TO_REAL, {discrete});
  auto normal = graph.add_distribution( // node_id = 5
      DistributionType::NORMAL,
      AtomicType::REAL,
      {real_discrete, stddev});

  return {discrete, normal};
}

std::tuple<NodeID, NodeID, NodeID> make_base_graph(Graph& graph) {
  // discrete = sample(Bernoulli(0.4))
  // real_discrete = to_real(discrete)
  // x = sample(Normal(real_discrete, 0.5))
  // // include descendants to make sure they are processed correctly
  // y = sample(Normal(x, stddev))
  // z = x + real_discrete
  // query(x)
  // observe(y, 3.1)
  //
  // Expected resulting graph:
  //
  // x = sample(bimixture(0.4, Normal(to_real(true), 1.0),
  //                           Normal(to_real(false), 1.0)))
  // y = sample(Normal(x, stddev))
  // z = x + real_discrete
  // query(x)
  // observe(y, 3.1)
  auto point_four = graph.add_constant_probability(0.4);
  auto stddev = graph.add_constant_pos_real(1.0);
  auto bernoulli = graph.add_distribution(
      DistributionType::BERNOULLI, AtomicType::BOOLEAN, {point_four});
  auto discrete =
      graph.add_operator(OperatorType::SAMPLE, {bernoulli}); // node_id = 3
  auto real_discrete = graph.add_operator(OperatorType::TO_REAL, {discrete});
  auto normal = graph.add_distribution( // node_id = 5
      DistributionType::NORMAL,
      AtomicType::REAL,
      {real_discrete, stddev});
  auto x = graph.add_operator(OperatorType::SAMPLE, {normal});
  auto normal2 = graph.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {x, stddev});
  auto y = graph.add_operator(OperatorType::SAMPLE, {normal2});
  auto z = graph.add_operator(OperatorType::ADD, {y, real_discrete});
  graph.query(x);
  graph.observe(y, 3.1);

  return {discrete, normal, y};
}

TEST(marginalization_extensional_test, marginalization) {
  Graph graph;
  NodeID discrete, normal, y;
  std::tie(discrete, normal, y) = make_base_graph(graph);

  cout << "Original:\n" << graph.to_string() << endl;

  auto expected = R"(0: CONSTANT(probability 0.4) (out nodes: 2)
1: CONSTANT(positive real 1) (out nodes: 5, 22, 11, 13)
2: BERNOULLI(0) (out nodes: 3, 8, 9)
3: SAMPLE(2) (out nodes: 4)
4: TO_REAL(3) (out nodes: 5, 24)
5: NORMAL(4, 1) (out nodes: )
6: CONSTANT(boolean 0) (out nodes: 8, 10)
7: CONSTANT(boolean 1) (out nodes: 9, 12)
8: LOG_PROB(2, 6) (out nodes: 14, 17)
9: LOG_PROB(2, 7) (out nodes: 14)
10: TO_REAL(6) (out nodes: 11)
11: NORMAL(10, 1) (out nodes: 20)
12: TO_REAL(7) (out nodes: 13)
13: NORMAL(12, 1) (out nodes: 20)
14: LOGSUMEXP(8, 9) (out nodes: 16)
15: CONSTANT(real -1) (out nodes: 16)
16: MULTIPLY(15, 14) (out nodes: 17)
17: ADD(8, 16) (out nodes: 18)
18: EXP(17) (out nodes: 19)
19: TO_PROBABILITY(18) (out nodes: 20)
20: BIMIXTURE(19, 11, 13) (out nodes: 21)
21: SAMPLE(20) (out nodes: 22) queried
22: NORMAL(21, 1) (out nodes: 23)
23: SAMPLE(22) (out nodes: 24) observed to be real 3.1
24: ADD(23, 4) (out nodes: )
)";

  marginalize(graph.get_node(discrete), graph);

  cout << "Result:\n" << graph.to_string() << endl;

  ASSERT_EQ(expected, graph.to_string());
}

TEST(marginalization_extensional_test, marginalization_on_non_sample_node) {
  Graph graph;
  NodeID discrete, normal, y;
  std::tie(discrete, normal, y) = make_base_graph(graph);

  EXPECT_THROW(
      marginalize(graph.get_node(normal), graph),
      marginalization_on_non_sample_node);
}

TEST(
    marginalization_extensional_test,
    marginalization_on_observed_or_queried_node) {
  Graph graph;
  NodeID discrete, normal, y;
  std::tie(discrete, normal, y) = make_base_graph(graph);
  graph.observe(discrete, true);

  EXPECT_THROW(
      marginalize(graph.get_node(discrete), graph),
      marginalization_on_observed_or_queried_node);
}

TEST(
    marginalization_extensional_test,
    marginalization_not_supported_for_samples_of) {
  Graph graph;
  auto point_four = graph.add_constant_pos_real(0.4);
  auto poisson = graph.add_distribution(
      DistributionType::POISSON, AtomicType::NATURAL, {point_four});
  auto x = graph.add_operator(OperatorType::SAMPLE, {poisson});

  EXPECT_THROW(
      marginalize(graph.get_node(x), graph),
      marginalization_not_supported_for_samples_of);
}

TEST(marginalization_extensional_test, marginalization_multiple_children) {
  Graph graph;
  NodeID discrete, normal, y;
  std::tie(discrete, normal, y) = make_base_graph(graph);

  // Algorithm does not currently support multiple children
  auto x2 = graph.add_operator(OperatorType::SAMPLE, {normal});

  EXPECT_THROW(
      marginalize(graph.get_node(discrete), graph),
      marginalization_multiple_children);
}

TEST(marginalization_extensional_test, marginalization_no_children) {
  Graph graph;
  NodeID discrete, normal;
  std::tie(discrete, normal) = make_childless_base_graph(graph);

  EXPECT_THROW(
      marginalize(graph.get_node(discrete), graph),
      marginalization_no_children);
}

TEST(marginalization_extensional_test, marginalization_inference) {
  Graph graph;
  NodeID discrete, normal, y;
  std::tie(discrete, normal, y) = make_base_graph(graph);

  auto num_samples = 50000;
  auto seed = std::time(nullptr);
  auto original_mean = graph.infer_mean(num_samples, InferenceType::NMC, seed);
  cout << "Mean from original graph: " << util::join(original_mean) << endl;

  marginalize(graph.get_node(discrete), graph);
  auto marginalized_mean =
      graph.infer_mean(num_samples, InferenceType::NMC, seed);
  cout << "Mean from marginalized graph: " << util::join(marginalized_mean)
       << endl;

  ASSERT_NEAR(original_mean[0], marginalized_mean[0], 1e-2);
}
