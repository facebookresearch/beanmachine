/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/marginalization/marginalized_graph.h"

using namespace beanmachine;
using namespace graph;

TEST(testmarginal, only_discrete) {
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

  MarginalizedGraph mgraph = MarginalizedGraph(g);
  mgraph.marginalize(2);
}

TEST(testmarginal, parent_and_child) {
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

  MarginalizedGraph mgraph = MarginalizedGraph(g);
  mgraph.marginalize(coin);
}

TEST(testmarginals, parent_and_children) {
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
  uint binomial_n = g.add_constant((natural_t)10);
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

  uint five = g.add_constant(5.0);
  uint binomial_plus_five =
      g.add_operator(OperatorType::ADD, {binomial_real, five});
  uint normal_2 = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {binomial_plus_five, one});
  uint n2 = g.add_operator(OperatorType::SAMPLE, {normal_2});
  g.query(n2);

  MarginalizedGraph mgraph = MarginalizedGraph(g);
  mgraph.marginalize(binomial_sample);
}
