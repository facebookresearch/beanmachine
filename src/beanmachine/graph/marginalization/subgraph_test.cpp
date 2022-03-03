/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <stdexcept>

#include "beanmachine/graph/marginalization/subgraph.h"

using namespace beanmachine;
using namespace graph;

TEST(testmarginal, subgraph_basic) {
  Graph g;
  uint half = g.add_constant_probability(0.5);
  uint bernoulli = g.add_distribution(
      DistributionType::BERNOULLI, AtomicType::BOOLEAN, {half});
  uint bernoulli_sample = g.add_operator(OperatorType::SAMPLE, {bernoulli});
  g.add_operator(OperatorType::TO_REAL, {bernoulli_sample});

  SubGraph subgraph = SubGraph(g);
  subgraph.add_node_by_id(bernoulli);
  subgraph.add_node_by_id(bernoulli_sample);
  subgraph.move_nodes_from_graph();

  Node* subgraph_bernoulli = subgraph.get_node(0);
  EXPECT_EQ(subgraph_bernoulli->node_type, NodeType::DISTRIBUTION);
  Node* subgraph_bernoulli_sample = subgraph.get_node(1);
  EXPECT_EQ(subgraph_bernoulli_sample->node_type, NodeType::OPERATOR);

  Node* graph_half = g.get_node(0);
  EXPECT_EQ(graph_half->node_type, NodeType::CONSTANT);
  Node* graph_to_real = g.get_node(1);
  EXPECT_EQ(graph_to_real->node_type, NodeType::OPERATOR);
  EXPECT_THROW(g.get_node(2), std::out_of_range);
}
