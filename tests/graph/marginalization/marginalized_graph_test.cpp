/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <stdexcept>

#include "beanmachine/graph/distribution/dummy_marginal.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/marginalization/marginalized_graph.h"
#include "beanmachine/graph/marginalization/subgraph.h"
#include "beanmachine/graph/operator/operator.h"

using namespace beanmachine;
using namespace graph;

TEST(testmarginal, DISABLED_only_discrete) {
  /*
  Original graph
  digraph G {
    "0: half" -> "1: bernoulli"
    "1: bernoulli" -> "2: bernoulli_sample\nquery\n(marginalize)"
  }

  Marginalized graph
  digraph G {
    subgraph cluster_0 {
      "0: half" -> "1: marginalized_distribution";
      label = "graph";
    }

    subgraph cluster_1 {
      "0: COPY half" -> "1: bernoulli"
      "1: bernoulli" -> "2: bernoulli_sample"
      label = "subgraph";
    }
  }
  */
  Graph g;
  uint half = g.add_constant_probability(0.5);
  uint bernoulli = g.add_distribution(
      DistributionType::BERNOULLI, AtomicType::BOOLEAN, {half});
  uint bernoulli_sample = g.add_operator(OperatorType::SAMPLE, {bernoulli});
  g.query(bernoulli_sample);

  marginalize_graph(g, 2);

  // check graph nodes
  Node* half_node = g.get_node(0);
  EXPECT_NEAR(half_node->value._double, 0.5, 1e-4);
  Node* marginalized_node = g.get_node(1);
  EXPECT_EQ(marginalized_node->node_type, NodeType::DISTRIBUTION);
  EXPECT_THROW(g.get_node(2), std::out_of_range);
  // check graph relationships
  EXPECT_EQ(half_node->out_nodes[0], marginalized_node);
  EXPECT_EQ(marginalized_node->in_nodes[0], half_node);

  // check subgraph nodes
  distribution::DummyMarginal* marginalized_distribution =
      dynamic_cast<distribution::DummyMarginal*>(marginalized_node);
  SubGraph* subgraph = marginalized_distribution->subgraph_ptr.get();
  Node* copy_half_node = subgraph->get_node(0);
  EXPECT_NEAR(copy_half_node->value._double, 0.5, 1e-4);
  Node* bernoulli_node = subgraph->get_node(1);
  EXPECT_EQ(bernoulli_node->node_type, NodeType::DISTRIBUTION);
  Node* bernoulli_sample_node = subgraph->get_node(2);
  EXPECT_EQ(bernoulli_sample_node->node_type, NodeType::OPERATOR);
  EXPECT_THROW(subgraph->get_node(3), std::out_of_range);
  // check subgraph relationships
  EXPECT_EQ(copy_half_node->out_nodes[0], bernoulli_node);
  EXPECT_EQ(bernoulli_node->in_nodes[0], copy_half_node);
  EXPECT_EQ(bernoulli_node->out_nodes[0], bernoulli_sample_node);
  EXPECT_EQ(bernoulli_sample_node->in_nodes[0], bernoulli_node);
}

TEST(testmarginal, DISABLED_parent_and_child) {
  /*
  Original graph:
  digraph G {
    "0: half" -> "1: bernoulli"
    "1: bernoulli" -> "2: coin\nmarginalize"
    "2: coin\nmarginalize" -> "3: coin_real"
    "3: coin_real" -> "5: normal"
    "4: one" -> "5: normal"
    "5: normal" -> "6: n\nquery"
  }

  Marginalized graph:
  digraph G {
    subgraph cluster_0 {
        "0: half" -> "2: marginalized_distribution"
        "1: one" -> "2: marginalized_distribution"
        "2: marginalized_distribution" -> "3: COPY n"
        label = "graph";
    }

    subgraph cluster_1 {
        label = "subgraph";
        "0: COPY half" -> "2: bernoulli"
        "1: COPY one" -> "5: normal"
        "2: bernoulli" -> "3: coin"
        "3: coin" -> "4: coin_real"
        "4: coin_real" -> "5: normal"
        "5: normal" -> "6: n"
    }
  }
  */
  Graph g;
  uint half = g.add_constant_probability(0.5);
  uint bernoulli = g.add_distribution(
      DistributionType::BERNOULLI, AtomicType::BOOLEAN, {half});
  uint coin = g.add_operator(OperatorType::SAMPLE, {bernoulli});
  uint coin_real = g.add_operator(OperatorType::TO_REAL, {coin});
  uint one = g.add_constant_pos_real(1.0);
  uint normal = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {coin_real, one});
  uint n = g.add_operator(OperatorType::SAMPLE, {normal});
  g.query(n);

  marginalize_graph(g, coin);

  // check graph nodes
  Node* half_node = g.get_node(0);
  EXPECT_NEAR(half_node->value._double, 0.5, 1e-4);
  Node* one_node = g.get_node(1);
  EXPECT_NEAR(one_node->value._double, 1.0, 1e-4);
  Node* marginalized_node = g.get_node(2);
  EXPECT_EQ(marginalized_node->node_type, NodeType::DISTRIBUTION);
  Node* sample_n_node = g.get_node(3);
  EXPECT_EQ(sample_n_node->node_type, NodeType::OPERATOR);
  EXPECT_THROW(g.get_node(4), std::out_of_range);
  // check graph relationships
  EXPECT_EQ(half_node->out_nodes[0], marginalized_node);
  EXPECT_EQ(marginalized_node->in_nodes[0], half_node);
  EXPECT_EQ(one_node->out_nodes[0], marginalized_node);
  EXPECT_EQ(marginalized_node->in_nodes[1], one_node);
  EXPECT_EQ(marginalized_node->out_nodes[0], sample_n_node);
  EXPECT_EQ(sample_n_node->in_nodes[0], marginalized_node);

  // check subgraph nodes
  distribution::DummyMarginal* marginalized_distribution =
      dynamic_cast<distribution::DummyMarginal*>(marginalized_node);
  SubGraph* subgraph = marginalized_distribution->subgraph_ptr.get();
  Node* copy_half_node = subgraph->get_node(0);
  EXPECT_NEAR(copy_half_node->value._double, 0.5, 1e-4);
  EXPECT_EQ(copy_half_node->node_type, NodeType::CONSTANT);
  Node* copy_one_node = subgraph->get_node(1);
  EXPECT_NEAR(copy_one_node->value._double, 1.0, 1e-4);
  EXPECT_EQ(copy_one_node->node_type, NodeType::CONSTANT);
  Node* bernoulli_node = subgraph->get_node(2);
  EXPECT_EQ(bernoulli_node->node_type, NodeType::DISTRIBUTION);
  Node* coin_node = subgraph->get_node(3);
  EXPECT_EQ(coin_node->node_type, NodeType::OPERATOR);
  Node* coin_real_node = subgraph->get_node(4);
  EXPECT_EQ(coin_real_node->node_type, NodeType::OPERATOR);
  Node* normal_node = subgraph->get_node(5);
  EXPECT_EQ(normal_node->node_type, NodeType::DISTRIBUTION);
  Node* n_node = subgraph->get_node(6);
  EXPECT_EQ(n_node->node_type, NodeType::OPERATOR);
  EXPECT_THROW(subgraph->get_node(7), std::out_of_range);
  // check subgraph relationships
  EXPECT_EQ(copy_half_node->out_nodes[0], bernoulli_node);
  EXPECT_EQ(bernoulli_node->in_nodes[0], copy_half_node);
  EXPECT_EQ(bernoulli_node->out_nodes[0], coin_node);
  EXPECT_EQ(coin_node->in_nodes[0], bernoulli_node);
  EXPECT_EQ(coin_node->out_nodes[0], coin_real_node);
  EXPECT_EQ(coin_real_node->in_nodes[0], coin_node);
  EXPECT_EQ(coin_real_node->out_nodes[0], normal_node);
  EXPECT_EQ(normal_node->in_nodes[0], coin_real_node);
  EXPECT_EQ(copy_one_node->out_nodes[0], normal_node);
  EXPECT_EQ(normal_node->in_nodes[1], copy_one_node);
  EXPECT_EQ(normal_node->out_nodes[0], n_node);
  EXPECT_EQ(n_node->in_nodes[0], normal_node);
}

TEST(testmarginals, DISABLED_parent_and_children) {
  /*
  Original graph:
  digraph G {
    subgraph cluster_1 {
        label = "subgraph";
        "2: binomial";
        "3: binomial_sample";
        "4: binomial_real";
        "6: normal_1";
        "9: binomial_plus_five";
        "10: normal_2";
    }
    "0: p" -> "2: binomial"
    "1: binomial_n" -> "2: binomial"
    "2: binomial" -> "3: binomial_sample"
    "3: binomial_sample" -> "4: binomial_real"
    "4: binomial_real" -> "6: normal_1"
    "5: one" -> "6: normal_1"
    "6: normal_1" -> "7: n1"
    "8: five" -> "9: binomial_plus_five"
    "4: binomial_real" -> "9: binomial_plus_five"
    "5: one" -> "10: normal_2"
    "9: binomial_plus_five" -> "10: normal_2"
    "10: normal_2" -> "11: n2"
  }

  Marginalized graph:
  digraph G {
    subgraph cluster_0 {
        "0: binomial_n" -> "4: marginalized_distribution"
        "1: p" -> "4: marginalized_distribution"
        "2: one" -> "4: marginalized_distribution"
        "3: five" -> "4: marginalized_distribution"
        "4: marginalized_distribution" -> "5: COPY n1"
        "4: marginalized_distribution" -> "6: COPY n2"
        label = "graph";
    }
    subgraph cluster_1 {
        "0: COPY binomial_n" -> "4: binomial"
        "1: COPY p" -> "4: binomial"
        "2: COPY one" -> "7: normal_1"
        "2: COPY one" -> "10: normal_2"
        "3: COPY five" -> "9: binomial_plus_five"
        "4: binomial" -> "5: binomial_sample"
        "5: binomial_sample" -> "6: binomial_real"
        "6: binomial_real" -> "7: normal_1"
        "7: normal_1" -> "8: n1"
        "6: binomial_real" -> "9: binomial_plus_five"
        "9: binomial_plus_five" -> "10: normal_2"
        "10: normal_2" -> "11: n2"
        label = "subgraph";
    }
}
  */
  Graph g;
  uint binomial_n = g.add_constant_natural(10);
  uint p = g.add_constant_probability(0.3);
  uint binomial = g.add_distribution(
      DistributionType::BINOMIAL, AtomicType::NATURAL, {binomial_n, p});
  uint binomial_sample = g.add_operator(OperatorType::SAMPLE, {binomial});
  uint binomial_real = g.add_operator(OperatorType::TO_REAL, {binomial_sample});

  uint one = g.add_constant_pos_real(1.0);
  uint normal_1 = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {binomial_real, one});
  uint n1 = g.add_operator(OperatorType::SAMPLE, {normal_1});
  g.query(n1);

  uint five = g.add_constant_real(5.0);
  uint binomial_plus_five =
      g.add_operator(OperatorType::ADD, {binomial_real, five});
  uint normal_2 = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {binomial_plus_five, one});
  uint n2 = g.add_operator(OperatorType::SAMPLE, {normal_2});
  g.query(n2);

  marginalize_graph(g, binomial_sample);

  // check graph nodes
  Node* binomial_n_node = g.get_node(0);
  EXPECT_EQ(binomial_n_node->value._natural, 10);
  Node* p_node = g.get_node(1);
  EXPECT_NEAR(p_node->value._double, 0.3, 1e-4);
  Node* one_node = g.get_node(2);
  EXPECT_NEAR(one_node->value._double, 1.0, 1e-4);
  Node* five_node = g.get_node(3);
  EXPECT_NEAR(five_node->value._double, 5.0, 1e-4);
  Node* marginalized_node = g.get_node(4);
  EXPECT_EQ(marginalized_node->node_type, NodeType::DISTRIBUTION);
  Node* sample_n1_node = g.get_node(5);
  EXPECT_EQ(sample_n1_node->node_type, NodeType::OPERATOR);
  Node* sample_n2_node = g.get_node(6);
  EXPECT_EQ(sample_n2_node->node_type, NodeType::OPERATOR);
  EXPECT_THROW(g.get_node(7), std::out_of_range);
  // check graph relationships
  EXPECT_EQ(binomial_n_node->out_nodes[0], marginalized_node);
  EXPECT_EQ(marginalized_node->in_nodes[0], binomial_n_node);
  EXPECT_EQ(p_node->out_nodes[0], marginalized_node);
  EXPECT_EQ(marginalized_node->in_nodes[1], p_node);
  EXPECT_EQ(one_node->out_nodes[0], marginalized_node);
  EXPECT_EQ(marginalized_node->in_nodes[2], one_node);
  EXPECT_EQ(five_node->out_nodes[0], marginalized_node);
  EXPECT_EQ(marginalized_node->in_nodes[3], five_node);
  EXPECT_EQ(marginalized_node->out_nodes[0], sample_n1_node);
  EXPECT_EQ(sample_n1_node->in_nodes[0], marginalized_node);
  EXPECT_EQ(marginalized_node->out_nodes[1], sample_n2_node);
  EXPECT_EQ(sample_n2_node->in_nodes[0], marginalized_node);

  // check subgraph nodes
  distribution::DummyMarginal* marginalized_distribution =
      dynamic_cast<distribution::DummyMarginal*>(marginalized_node);
  SubGraph* subgraph = marginalized_distribution->subgraph_ptr.get();
  Node* copy_binomial_n_node = subgraph->get_node(0);
  EXPECT_EQ(copy_binomial_n_node->value._natural, 10);
  EXPECT_EQ(copy_binomial_n_node->node_type, NodeType::CONSTANT);
  Node* copy_p_node = subgraph->get_node(1);
  EXPECT_NEAR(copy_p_node->value._double, 0.3, 1e-4);
  EXPECT_EQ(copy_p_node->node_type, NodeType::CONSTANT);
  Node* copy_one_node = subgraph->get_node(2);
  EXPECT_NEAR(copy_one_node->value._double, 1.0, 1e-4);
  EXPECT_EQ(copy_one_node->node_type, NodeType::CONSTANT);
  Node* copy_five_node = subgraph->get_node(3);
  EXPECT_NEAR(copy_five_node->value._double, 5.0, 1e-4);
  EXPECT_EQ(copy_five_node->node_type, NodeType::CONSTANT);
  Node* binomial_node = subgraph->get_node(4);
  EXPECT_EQ(binomial_node->node_type, NodeType::DISTRIBUTION);
  Node* binomial_sample_node = subgraph->get_node(5);
  EXPECT_EQ(binomial_sample_node->node_type, NodeType::OPERATOR);
  Node* binomial_real_node = subgraph->get_node(6);
  EXPECT_EQ(binomial_real_node->node_type, NodeType::OPERATOR);
  Node* normal_1_node = subgraph->get_node(7);
  EXPECT_EQ(normal_1_node->node_type, NodeType::DISTRIBUTION);
  Node* n1_node = subgraph->get_node(8);
  EXPECT_EQ(n1_node->node_type, NodeType::OPERATOR);
  Node* binomial_plus_five_node = subgraph->get_node(9);
  EXPECT_EQ(binomial_plus_five_node->node_type, NodeType::OPERATOR);
  Node* normal_2_node = subgraph->get_node(10);
  EXPECT_EQ(normal_1_node->node_type, NodeType::DISTRIBUTION);
  Node* n2_node = subgraph->get_node(11);
  EXPECT_EQ(n1_node->node_type, NodeType::OPERATOR);
  EXPECT_THROW(subgraph->get_node(12), std::out_of_range);
  // check subgraph relationships
  EXPECT_EQ(copy_binomial_n_node->out_nodes[0], binomial_node);
  EXPECT_EQ(binomial_node->in_nodes[0], copy_binomial_n_node);
  EXPECT_EQ(copy_p_node->out_nodes[0], binomial_node);
  EXPECT_EQ(binomial_node->in_nodes[1], copy_p_node);
  EXPECT_EQ(copy_one_node->out_nodes[0], normal_1_node);
  EXPECT_EQ(normal_1_node->in_nodes[1], copy_one_node);
  EXPECT_EQ(copy_one_node->out_nodes[1], normal_2_node);
  EXPECT_EQ(normal_2_node->in_nodes[1], copy_one_node);
  EXPECT_EQ(copy_five_node->out_nodes[0], binomial_plus_five_node);
  EXPECT_EQ(binomial_plus_five_node->in_nodes[1], copy_five_node);
  EXPECT_EQ(binomial_sample_node->in_nodes[0], binomial_node);
  EXPECT_EQ(binomial_node->out_nodes[0], binomial_sample_node);
  EXPECT_EQ(binomial_real_node->in_nodes[0], binomial_sample_node);
  EXPECT_EQ(binomial_sample_node->out_nodes[0], binomial_real_node);
  EXPECT_EQ(normal_1_node->in_nodes[0], binomial_real_node);
  EXPECT_EQ(binomial_real_node->out_nodes[0], normal_1_node);
  EXPECT_EQ(n1_node->in_nodes[0], normal_1_node);
  EXPECT_EQ(normal_1_node->out_nodes[0], n1_node);
  EXPECT_EQ(binomial_plus_five_node->in_nodes[0], binomial_real_node);
  EXPECT_EQ(binomial_real_node->out_nodes[1], binomial_plus_five_node);
  EXPECT_EQ(normal_2_node->in_nodes[0], binomial_plus_five_node);
  EXPECT_EQ(binomial_plus_five_node->out_nodes[0], normal_2_node);
  EXPECT_EQ(n2_node->in_nodes[0], normal_2_node);
  EXPECT_EQ(normal_2_node->out_nodes[0], n2_node);
}

TEST(testmarginal, DISABLED_parent_and_long_child) {
  /*
  Original graph:
  digraph G {
    "0: half" -> "1: bernoulli"
    "1: bernoulli" -> "2: coin\nmarginalize"
    "2: coin\nmarginalize" -> "3: coin_real"
    "3: coin_real" -> "5: normal_1"
    "4: one" -> "5: normal_1"
    "5: normal_1" -> "6: n1"
    "6: n1" -> "7: normal_2"
    "4: one" -> "7: normal_2"
    "7: normal_2" -> "8:n2 \nquery"
  }

  Marginalized graph:
  digraph G {
    subgraph cluster_0 {
        "0: half" -> "2: marginalized_distribution"
        "1: one" -> "2: marginalized_distribution"
        "2: marginalized_distribution" -> "3: COPY n1"
        "3: COPY n1" -> "4: normal_2"
        "4: normal_2" -> "5: n2"
        "1: one" -> "4: normal_2"
        label = "graph";
    }

    subgraph cluster_1 {
        label = "subgraph";
        "0: COPY half" -> "2: bernoulli"
        "1: COPY one" -> "5: normal"
        "2: bernoulli" -> "3: coin"
        "3: coin" -> "4: coin_real"
        "4: coin_real" -> "5: normal"
        "5: normal" -> "6: n"
    }
  }
  */
  Graph g;
  uint half = g.add_constant_probability(0.5);
  uint bernoulli = g.add_distribution(
      DistributionType::BERNOULLI, AtomicType::BOOLEAN, {half});
  uint coin = g.add_operator(OperatorType::SAMPLE, {bernoulli});
  uint coin_real = g.add_operator(OperatorType::TO_REAL, {coin});
  uint one = g.add_constant_pos_real(1.0);
  uint normal_1 = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {coin_real, one});
  uint n1 = g.add_operator(OperatorType::SAMPLE, {normal_1});
  uint normal_2 =
      g.add_distribution(DistributionType::NORMAL, AtomicType::REAL, {n1, one});
  uint n2 = g.add_operator(OperatorType::SAMPLE, {normal_2});
  g.query(n2);

  marginalize_graph(g, coin);

  // check graph nodes
  Node* half_node = g.get_node(0);
  EXPECT_NEAR(half_node->value._double, 0.5, 1e-4);
  Node* one_node = g.get_node(1);
  EXPECT_NEAR(one_node->value._double, 1.0, 1e-4);
  Node* marginalized_node = g.get_node(2);
  EXPECT_EQ(marginalized_node->node_type, NodeType::DISTRIBUTION);
  Node* sample_n_node = g.get_node(3);
  EXPECT_EQ(sample_n_node->node_type, NodeType::OPERATOR);
  Node* normal_2_node = g.get_node(4);
  EXPECT_EQ(normal_2_node->node_type, NodeType::DISTRIBUTION);
  Node* n2_node = g.get_node(5);
  EXPECT_EQ(n2_node->node_type, NodeType::OPERATOR);
  EXPECT_THROW(g.get_node(6), std::out_of_range);
  // check graph relationships
  EXPECT_EQ(half_node->out_nodes[0], marginalized_node);
  EXPECT_EQ(marginalized_node->in_nodes[0], half_node);
  EXPECT_EQ(one_node->out_nodes[1], marginalized_node);
  EXPECT_EQ(marginalized_node->in_nodes[1], one_node);
  EXPECT_EQ(marginalized_node->out_nodes[0], sample_n_node);
  EXPECT_EQ(sample_n_node->in_nodes[0], marginalized_node);
  EXPECT_EQ(sample_n_node->out_nodes[0], normal_2_node);
  EXPECT_EQ(normal_2_node->in_nodes[0], sample_n_node);
  EXPECT_EQ(one_node->out_nodes[0], normal_2_node);
  EXPECT_EQ(normal_2_node->in_nodes[1], one_node);
  EXPECT_EQ(normal_2_node->out_nodes[0], n2_node);
  EXPECT_EQ(n2_node->in_nodes[0], normal_2_node);

  // check subgraph nodes
  distribution::DummyMarginal* marginalized_distribution =
      dynamic_cast<distribution::DummyMarginal*>(marginalized_node);
  SubGraph* subgraph = marginalized_distribution->subgraph_ptr.get();
  Node* copy_half_node = subgraph->get_node(0);
  EXPECT_NEAR(copy_half_node->value._double, 0.5, 1e-4);
  EXPECT_EQ(copy_half_node->node_type, NodeType::CONSTANT);
  Node* copy_one_node = subgraph->get_node(1);
  EXPECT_NEAR(copy_one_node->value._double, 1.0, 1e-4);
  EXPECT_EQ(copy_one_node->node_type, NodeType::CONSTANT);
  Node* bernoulli_node = subgraph->get_node(2);
  EXPECT_EQ(bernoulli_node->node_type, NodeType::DISTRIBUTION);
  Node* coin_node = subgraph->get_node(3);
  EXPECT_EQ(coin_node->node_type, NodeType::OPERATOR);
  Node* coin_real_node = subgraph->get_node(4);
  EXPECT_EQ(coin_real_node->node_type, NodeType::OPERATOR);
  Node* normal_node = subgraph->get_node(5);
  EXPECT_EQ(normal_node->node_type, NodeType::DISTRIBUTION);
  Node* n_node = subgraph->get_node(6);
  EXPECT_EQ(n_node->node_type, NodeType::OPERATOR);
  EXPECT_THROW(subgraph->get_node(7), std::out_of_range);
  // check subgraph relationships
  EXPECT_EQ(copy_half_node->out_nodes[0], bernoulli_node);
  EXPECT_EQ(bernoulli_node->in_nodes[0], copy_half_node);
  EXPECT_EQ(bernoulli_node->out_nodes[0], coin_node);
  EXPECT_EQ(coin_node->in_nodes[0], bernoulli_node);
  EXPECT_EQ(coin_node->out_nodes[0], coin_real_node);
  EXPECT_EQ(coin_real_node->in_nodes[0], coin_node);
  EXPECT_EQ(coin_real_node->out_nodes[0], normal_node);
  EXPECT_EQ(normal_node->in_nodes[0], coin_real_node);
  EXPECT_EQ(copy_one_node->out_nodes[0], normal_node);
  EXPECT_EQ(normal_node->in_nodes[1], copy_one_node);
  EXPECT_EQ(normal_node->out_nodes[0], n_node);
  EXPECT_EQ(n_node->in_nodes[0], normal_node);
}

TEST(testmarginal, DISABLED_upstream_from_parents) {
  /*
  Original graph:
  digraph G {
    "0: half" -> "1: bernoulli"
    "1: bernoulli" -> "2: coin\nmarginalize"
    "2: coin\nmarginalize" -> "3: coin_real"
    "3: coin_real" -> "7: normal"
    "4: one" -> "6: multiply"
    "5: two" -> "6: multiply"
    "6: multiply" -> "7: normal"
    "7: normal" -> "8: n\nquery"
  }

  Marginalized graph:
  digraph G {
    subgraph cluster_0 {
        "0: half" -> "4: marginalized_distribution"
        "1: one" -> "3: multiply"
        "2: two" -> "3: multiply"
        "3: multiply" -> "4: marginalized_distribution"
        "4: marginalized_distribution" -> "5: COPY n"
        label = "graph";
    }

    subgraph cluster_1 {
        label = "subgraph";
        "0: COPY half" -> "2: bernoulli"
        "1: COPY multiply" -> "5: normal"
        "2: bernoulli" -> "3: coin"
        "3: coin" -> "4: coin_real"
        "4: coin_real" -> "5: normal"
        "5: normal" -> "6: n"
    }
  }
  */
  Graph g;
  uint half = g.add_constant_probability(0.5);
  uint bernoulli = g.add_distribution(
      DistributionType::BERNOULLI, AtomicType::BOOLEAN, {half});
  uint coin = g.add_operator(OperatorType::SAMPLE, {bernoulli});
  uint coin_real = g.add_operator(OperatorType::TO_REAL, {coin});
  uint one = g.add_constant_pos_real(1.0);
  uint two = g.add_constant_pos_real(2.0);
  uint multiply = g.add_operator(OperatorType::MULTIPLY, {one, two});
  uint normal = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {coin_real, multiply});
  uint n = g.add_operator(OperatorType::SAMPLE, {normal});
  g.query(n);

  marginalize_graph(g, coin);

  // check graph nodes
  Node* half_node = g.get_node(0);
  EXPECT_NEAR(half_node->value._double, 0.5, 1e-4);
  Node* one_node = g.get_node(1);
  EXPECT_NEAR(one_node->value._double, 1.0, 1e-4);
  Node* two_node = g.get_node(2);
  EXPECT_NEAR(two_node->value._double, 2.0, 1e-4);
  Node* multiply_node = g.get_node(3);
  EXPECT_EQ(multiply_node->node_type, NodeType::OPERATOR);
  Node* marginalized_node = g.get_node(4);
  EXPECT_EQ(marginalized_node->node_type, NodeType::DISTRIBUTION);
  Node* sample_n_node = g.get_node(5);
  EXPECT_EQ(sample_n_node->node_type, NodeType::OPERATOR);
  EXPECT_THROW(g.get_node(6), std::out_of_range);
  // check graph relationships
  EXPECT_EQ(half_node->out_nodes[0], marginalized_node);
  EXPECT_EQ(marginalized_node->in_nodes[0], half_node);
  EXPECT_EQ(one_node->out_nodes[0], multiply_node);
  EXPECT_EQ(multiply_node->in_nodes[0], one_node);
  EXPECT_EQ(two_node->out_nodes[0], multiply_node);
  EXPECT_EQ(multiply_node->in_nodes[1], two_node);
  EXPECT_EQ(multiply_node->out_nodes[0], marginalized_node);
  EXPECT_EQ(marginalized_node->in_nodes[1], multiply_node);
  EXPECT_EQ(marginalized_node->out_nodes[0], sample_n_node);
  EXPECT_EQ(sample_n_node->in_nodes[0], marginalized_node);

  // check subgraph nodes
  distribution::DummyMarginal* marginalized_distribution =
      dynamic_cast<distribution::DummyMarginal*>(marginalized_node);
  SubGraph* subgraph = marginalized_distribution->subgraph_ptr.get();
  Node* copy_half_node = subgraph->get_node(0);
  EXPECT_NEAR(copy_half_node->value._double, 0.5, 1e-4);
  EXPECT_EQ(copy_half_node->node_type, NodeType::CONSTANT);
  Node* copy_multiply_node = subgraph->get_node(1);
  EXPECT_EQ(copy_multiply_node->node_type, NodeType::CONSTANT);
  Node* bernoulli_node = subgraph->get_node(2);
  EXPECT_EQ(bernoulli_node->node_type, NodeType::DISTRIBUTION);
  Node* coin_node = subgraph->get_node(3);
  EXPECT_EQ(coin_node->node_type, NodeType::OPERATOR);
  Node* coin_real_node = subgraph->get_node(4);
  EXPECT_EQ(coin_real_node->node_type, NodeType::OPERATOR);
  Node* normal_node = subgraph->get_node(5);
  EXPECT_EQ(normal_node->node_type, NodeType::DISTRIBUTION);
  Node* n_node = subgraph->get_node(6);
  EXPECT_EQ(n_node->node_type, NodeType::OPERATOR);
  EXPECT_THROW(subgraph->get_node(7), std::out_of_range);
  // check subgraph relationships
  EXPECT_EQ(copy_half_node->out_nodes[0], bernoulli_node);
  EXPECT_EQ(bernoulli_node->in_nodes[0], copy_half_node);
  EXPECT_EQ(bernoulli_node->out_nodes[0], coin_node);
  EXPECT_EQ(coin_node->in_nodes[0], bernoulli_node);
  EXPECT_EQ(coin_node->out_nodes[0], coin_real_node);
  EXPECT_EQ(coin_real_node->in_nodes[0], coin_node);
  EXPECT_EQ(coin_real_node->out_nodes[0], normal_node);
  EXPECT_EQ(normal_node->in_nodes[0], coin_real_node);
  EXPECT_EQ(copy_multiply_node->out_nodes[0], normal_node);
  EXPECT_EQ(normal_node->in_nodes[1], copy_multiply_node);
  EXPECT_EQ(normal_node->out_nodes[0], n_node);
  EXPECT_EQ(n_node->in_nodes[0], normal_node);
}
