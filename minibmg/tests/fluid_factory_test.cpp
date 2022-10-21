/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <memory>
#include "beanmachine/minibmg/fluid_factory.h"
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/pretty.h"

using namespace ::testing;
using namespace beanmachine::minibmg;

namespace fluent_factory_test {

std::string raw_json = R"({
  "comment": "created by graph_to_json",
  "nodes": [
    {
      "operator": "CONSTANT",
      "sequence": 0,
      "value": 2
    },
    {
      "in_nodes": [
        0,
        0
      ],
      "operator": "DISTRIBUTION_BETA",
      "sequence": 1
    },
    {
      "in_nodes": [
        1
      ],
      "operator": "SAMPLE",
      "sequence": 2
    },
    {
      "in_nodes": [
        2
      ],
      "operator": "DISTRIBUTION_BERNOULLI",
      "sequence": 3
    },
    {
      "in_nodes": [
        3
      ],
      "operator": "SAMPLE",
      "sequence": 4
    },
    {
      "in_nodes": [
        3
      ],
      "operator": "SAMPLE",
      "sequence": 5
    },
    {
      "in_nodes": [
        3
      ],
      "operator": "SAMPLE",
      "sequence": 6
    },
    {
      "in_nodes": [
        3
      ],
      "operator": "SAMPLE",
      "sequence": 7
    },
    {
      "in_nodes": [
        3
      ],
      "operator": "SAMPLE",
      "sequence": 8
    }
  ],
  "observations": [
    {
      "node": 4,
      "value": 1
    },
    {
      "node": 5,
      "value": 1
    },
    {
      "node": 6,
      "value": 1
    },
    {
      "node": 7,
      "value": 0
    },
    {
      "node": 8,
      "value": 0
    }
  ],
  "queries": [
    2
  ]
})";

TEST(fluent_factory_test, simple_test) {
  Graph::FluidFactory fac;
  auto b = beta(2, 2);
  auto s = sample(b);
  auto r = bernoulli(s);

  fac.observe(sample(r), 1);
  fac.observe(sample(r), 1);
  fac.observe(sample(r), 1);
  fac.observe(sample(r), 0);
  fac.observe(sample(r), 0);
  fac.query(s);

  auto graph = fac.build();
  auto json = folly::toPrettyJson(graph_to_json(graph));
  ASSERT_EQ(raw_json, json);
}

TEST(fluent_factory_test, deduplication_01) {
  Graph::FluidFactory fac;
  auto b = beta(2, 2);
  auto s = sample(b, "S0");
  auto r = bernoulli(s);

  fac.observe(sample(r, "S1"), 1);
  fac.observe(sample(r, "S2"), 1);
  fac.observe(sample(r, "S3"), 1);
  fac.observe(sample(r, "S4"), 0);
  fac.observe(sample(r, "S5"), 0);
  fac.query(s);

  auto r2 = bernoulli(s);

  fac.observe(sample(r2, "S6"), 1);
  fac.observe(sample(r2, "S7"), 1);
  fac.observe(sample(r2, "S8"), 1);
  fac.observe(sample(r2, "S9"), 0);
  fac.observe(sample(r2, "S10"), 0);

  auto graph = fac.build();
  auto pretty = pretty_print(graph);
  auto expected = R"+(auto temp_1 = sample(beta(2, 2), "S0");
auto temp_2 = bernoulli(temp_1);
Graph::FluidFactory fac;
fac.query(temp_1);
fac.observe(sample(temp_2, "S1"), 1);
fac.observe(sample(temp_2, "S2"), 1);
fac.observe(sample(temp_2, "S3"), 1);
fac.observe(sample(temp_2, "S4"), 0);
fac.observe(sample(temp_2, "S5"), 0);
fac.observe(sample(temp_2, "S6"), 1);
fac.observe(sample(temp_2, "S7"), 1);
fac.observe(sample(temp_2, "S8"), 1);
fac.observe(sample(temp_2, "S9"), 0);
fac.observe(sample(temp_2, "S10"), 0);
)+";
  ASSERT_EQ(expected, pretty);
}

TEST(fluent_factory_test, deduplication_02) {
  Value final = 0;
  for (int i = 0; i < 2; i++) {
    auto t1 = beta(2, 2);
    auto t2 = sample(t1, "S0") + sample(t1, "S1");
    auto t3 = t2 + t2;
    auto t4 = t3 - t2;
    auto t5 = -(t4 + t3);
    auto t6 = t5 * t4;
    auto t7 = t6 / t5;
    auto t8 = pow(t7, t6);
    auto t9 = exp(t8 + t7);
    auto t10 = log(t9 + t8);
    auto t11 = atan(t10 + t9);
    auto t12 = lgamma(t11 + t10);
    auto t13 = polygamma(2, t11 + t12);
    auto t14 = if_equal(t13, t12, t11, t10);
    auto t15 = if_less(t14, t13, t12, t11);
    auto t16 = normal(t15, t14);
    auto t17 = sample(t16, "S2") + sample(t16, "S3");
    auto t18 = half_normal(t17);
    auto t19 = sample(t18, "S4") + sample(t18, "S5");
    // note using the same sample twice here.
    auto t20 = bernoulli(sample(t1, "S6"));
    auto t21 = bernoulli(sample(t1, "S6"));
    auto t22 = sample(t20, "S7") + sample(t20, "S8");
    auto t23 = t22 + t19 + t17 + t15;
    final = final + t23;
  }

  Graph::FluidFactory fac;
  fac.query(final);
  auto graph = fac.build();
  auto pretty = pretty_print(graph);
  auto expected = R"+(auto temp_1 = beta(2, 2);
auto temp_2 = bernoulli(sample(temp_1, "S6"));
auto temp_3 = sample(temp_1, "S0") + sample(temp_1, "S1");
auto temp_4 = temp_3 + temp_3;
auto temp_5 = temp_4 - temp_3;
auto temp_6 = -(temp_5 + temp_4);
auto temp_7 = temp_6 * temp_5;
auto temp_8 = temp_7 / temp_6;
auto temp_9 = pow(temp_8, temp_7);
auto temp_10 = exp(temp_9 + temp_8);
auto temp_11 = log(temp_10 + temp_9);
auto temp_12 = atan(temp_11 + temp_10);
auto temp_13 = lgamma(temp_12 + temp_11);
auto temp_14 = polygamma(2, temp_12 + temp_13);
auto temp_15 = if_equal(temp_14, temp_13, temp_12, temp_11);
auto temp_16 = if_less(temp_15, temp_14, temp_13, temp_12);
auto temp_17 = normal(temp_16, temp_15);
auto temp_18 = sample(temp_17, "S2") + sample(temp_17, "S3");
auto temp_19 = half_normal(temp_18);
auto temp_20 = sample(temp_2, "S7") + sample(temp_2, "S8") + (sample(temp_19, "S4") + sample(temp_19, "S5")) + temp_18 + temp_16;
Graph::FluidFactory fac;
fac.query(0 + temp_20 + temp_20);
)+";
  ASSERT_EQ(expected, pretty);
}

// test constant folding of log1p
TEST(fluent_factory_test, log1p_kfold) {
  Value k = 1.1;
  Value l = log1p(k);
  Graph::FluidFactory fac;
  auto qi = fac.query(l);
  Graph g = fac.build();
  ASSERT_EQ(g.size(), 2);
  Value folded = std::dynamic_pointer_cast<const ScalarNode>(opt(g.queries[0]));
  Value expected = std::log1p(1.1);
  // assert that the optimized graph is just the resulting constant.
  ASSERT_TRUE(NodepIdentityEquals{}(expected.node, folded.node));
}

} // namespace fluent_factory_test
