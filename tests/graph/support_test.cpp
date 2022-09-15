/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <vector>

#include <gtest/gtest.h>

#include <beanmachine/graph/graph.h>
#include <beanmachine/graph/support.h>

using namespace beanmachine;
using namespace graph;

TEST(testgraph, support) {
  graph::Graph g;
  // c1 and c2 are parents of o1 -> d1 -> o2
  uint c1 = g.add_constant_probability(.3);
  uint c2 = g.add_constant_probability(.4);
  uint o1 = g.add_operator(
      graph::OperatorType::MULTIPLY, std::vector<uint>({c1, c2}));
  uint d1 = g.add_distribution(
      graph::DistributionType::BERNOULLI,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>{o1});
  uint o2 =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d1}));
  // c3 and c4 are parents of o3 -> d2 -> o4
  uint c3 = g.add_constant_pos_real(.3);
  uint c4 = g.add_constant_pos_real(.4);
  uint o3 = g.add_operator(
      graph::OperatorType::MULTIPLY, std::vector<uint>({c3, c4}));
  uint d2 = g.add_distribution(
      graph::DistributionType::BERNOULLI_NOISY_OR,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>{o3});
  uint o4 =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d2}));
  uint ro2 =
      g.add_operator(graph::OperatorType::TO_POS_REAL, std::vector<uint>({o2}));
  uint ro4 =
      g.add_operator(graph::OperatorType::TO_POS_REAL, std::vector<uint>({o4}));
  // o2 and o4 -> o5
  uint o5 = g.add_operator(
      graph::OperatorType::MULTIPLY, std::vector<uint>({ro2, ro4}));
  uint d3 = g.add_distribution(
      graph::DistributionType::BERNOULLI_NOISY_OR,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({o5}));
  uint o6 =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d3}));
  uint o7 =
      g.add_operator(graph::OperatorType::TO_POS_REAL, std::vector<uint>({o6}));
  uint o8 = g.add_operator(
      graph::OperatorType::MULTIPLY, std::vector<uint>({o3, o7}));
  // observe o4 so support includes c3, c4, o3, d2, o4
  // however, we limit support to operators only: o3 and o4
  g.observe(o4, true);
  auto ordered_support_node_ids = g.compute_ordered_support_node_ids();
  EXPECT_EQ(ordered_support_node_ids.size(), 2);
  EXPECT_EQ(*ordered_support_node_ids.begin(), o3);
  EXPECT_EQ(*ordered_support_node_ids.rbegin(), o4);
  std::vector<uint> det_nodes;
  std::vector<uint> sto_nodes;
  std::tie(det_nodes, sto_nodes) =
      g.compute_affected_nodes(o3, ordered_support_node_ids);
  // o3 -> det: o3, d2, o8 sto: o4
  // limiting to operators: o3 -> det: o3, o8 sto: o4
  // limiting to support: c3 -> det: o3 sto: o4
  EXPECT_EQ(det_nodes.size(), 1);
  EXPECT_EQ(det_nodes.front(), o3);
  EXPECT_EQ(sto_nodes.size(), 1);
  EXPECT_EQ(sto_nodes.front(), o4);

  DeterministicAncestors det_anc;
  StochasticAncestors sto_anc;
  std::tie(det_anc, sto_anc) =
      collect_deterministic_and_stochastic_ancestors(g);
  // o4 -ancestors-> det: c3, c4, o3, d2 sto:
  // restricting to operators: det: o3 sto:
  EXPECT_EQ(det_anc[o4].size(), 1);
  EXPECT_EQ(det_anc[o4].front(), o3);
  EXPECT_EQ(sto_anc[o4].size(), 0);
  // restricting to operators, ancestors(o8) = det: o3, o7 sto: o6
  EXPECT_EQ(det_anc[o8].size(), 2);
  EXPECT_EQ(det_anc[o8].front(), o3);
  EXPECT_EQ(det_anc[o8].back(), o7);
  EXPECT_EQ(sto_anc[o8].size(), 1);
  EXPECT_EQ(sto_anc[o8].front(), o6);
  // ancestors(o5) = det: ro2, ro4, sto: o2 o4
  EXPECT_EQ(det_anc[o5].size(), 2);
  EXPECT_EQ(det_anc[o5].front(), ro2);
  EXPECT_EQ(det_anc[o5].back(), ro4);
  EXPECT_EQ(sto_anc[o5].size(), 2);
  EXPECT_EQ(sto_anc[o5].front(), o2);
  EXPECT_EQ(sto_anc[o5].back(), o4);

  // query o5 so support is now 13 nodes:
  //   c1, c2, o1, d1, o2, ro2, c3, c4, o3, d2, o4, ro5, o5
  // but we only include operators o1, o2, ro2, o3, o4, ro4, and o5
  g.query(o5);
  ordered_support_node_ids = g.compute_ordered_support_node_ids();
  EXPECT_EQ(ordered_support_node_ids.size(), 7);
  // o4 -> det: o5, d3 sto: o4, o6
  // limiting to operators: o4 -> det: ro4, o5 sto: o4, o6
  // note: o7 and o8 are not in the descendants of o4
  // because the descendant chain gets cut off at the stochastic node o6
  // limiting to support: o4 -> det: ro4, o5 sto: o4
  std::tie(det_nodes, sto_nodes) =
      g.compute_affected_nodes(o4, ordered_support_node_ids);
  EXPECT_EQ(det_nodes.size(), 2);
  EXPECT_EQ(det_nodes.front(), ro4);
  EXPECT_EQ(det_nodes.back(), o5);
  EXPECT_EQ(sto_nodes.size(), 1);
  EXPECT_EQ(sto_nodes.front(), o4);
}

TEST(testgraph, full_support) {
  /*
  Visualize graph
    digraph G {
    "p" -> "bernoulli"
    "bernoulli" -> "coin"
    "coin" -> "coin_real"
    "coin_real" -> "normal1"
    "one" -> "normal1"
    "normal1" -> "n1"
    "five" -> "coin_plus_five"
    "coin_real" -> "coin_plus_five"
    "coin_plus_five" -> "normal2"
    "one" -> "normal2"
    "normal2" -> "n2 (query)"
    }
  */
  // build graph
  Graph g;
  uint one = g.add_constant_pos_real(1.0);
  uint p = g.add_constant_probability(0.5);
  uint bernoulli =
      g.add_distribution(DistributionType::BERNOULLI, AtomicType::BOOLEAN, {p});
  uint coin = g.add_operator(OperatorType::SAMPLE, {bernoulli});
  g.query(coin);
  uint coin_real = g.add_operator(OperatorType::TO_REAL, {coin});
  // should not be in support since not observed or queried
  uint normal1 = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {coin_real, one});
  uint n1 = g.add_operator(OperatorType::SAMPLE, {normal1});
  // should be in support since queried
  uint five = g.add_constant(5.0);
  uint coin_plus_five = g.add_operator(OperatorType::ADD, {coin_real, five});
  uint normal2 = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {coin_plus_five, one});
  uint n2 = g.add_operator(OperatorType::SAMPLE, {normal2});
  g.query(n2);

  // begin tests

  // support includes all operators and factors up to observations and
  // queries doesn't include distributions or constants
  // TODO: not sure about factors since they are akin to distributions; check
  // this.
  std::set<uint> ordered_support_node_ids =
      g.compute_ordered_support_node_ids();
  std::set<uint> expected_support = {coin, coin_real, coin_plus_five, n2};
  EXPECT_EQ(ordered_support_node_ids, expected_support);
  EXPECT_EQ(ordered_support_node_ids.count(n1), 0);

  // full_support includes *all* nodes up to observations and queries
  // includes distributions and constants
  std::set<uint> full_support = g.compute_full_ordered_support_node_ids();
  std::set<uint> expected_full_support = {
      p, bernoulli, coin, coin_real, five, coin_plus_five, one, normal2, n2};
  EXPECT_EQ(full_support, expected_full_support);

  std::vector<uint> det_nodes;
  std::vector<uint> sto_nodes;
  // computes node and nodes up to stochastic children
  // deterministic nodes only have operator nodes
  // stochastic node includes root node
  std::tie(det_nodes, sto_nodes) =
      g.compute_affected_nodes(coin, ordered_support_node_ids);
  std::vector<uint> expected_det_nodes = {coin_real, coin_plus_five};
  std::vector<uint> expected_sto_nodes = {coin, n2};
  EXPECT_EQ(det_nodes, expected_det_nodes);
  EXPECT_EQ(sto_nodes, expected_sto_nodes);

  // computes nodes up to stochastic children
  // deterministic nodes consist of all nodes
  // (constants, distributions, operators, etc)
  // stochastic node only includes children
  std::tie(det_nodes, sto_nodes) = g.compute_children(coin, full_support);
  expected_det_nodes = {coin_real, coin_plus_five, normal2};
  expected_sto_nodes = {n2};
  EXPECT_EQ(det_nodes, expected_det_nodes);
  EXPECT_EQ(sto_nodes, expected_sto_nodes);
}
