/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "beanmachine/minibmg/minibmg.h"

using namespace ::testing;
using namespace ::beanmachine::minibmg;

std::string raw_json = R"({
  "comment": "created by graph_to_json",
  "queries": [ 2 ],
  "observations": [
    { "node": 4, "value": 1.0 },
    { "node": 5, "value": 1.0 },
    { "node": 6, "value": 1.0 },
    { "node": 7, "value": 0.0 }
  ],
  "nodes": [
    {
      "operator": "CONSTANT",
      "sequence": 0,
      "type": "REAL",
      "value": 2
    },
    {
      "in_nodes": [
        0,
        0
      ],
      "operator": "DISTRIBUTION_BETA",
      "sequence": 1,
      "type": "DISTRIBUTION"
    },
    {
      "in_nodes": [
        1
      ],
      "operator": "SAMPLE",
      "sequence": 2,
      "type": "REAL"
    },
    {
      "in_nodes": [
        2
      ],
      "operator": "DISTRIBUTION_BERNOULLI",
      "sequence": 3,
      "type": "DISTRIBUTION"
    },
    {
      "in_nodes": [
        3
      ],
      "operator": "SAMPLE",
      "sequence": 4,
      "type": "REAL"
    },
    {
      "in_nodes": [
        3
      ],
      "operator": "SAMPLE",
      "sequence": 5,
      "type": "REAL"
    },
    {
      "in_nodes": [
        3
      ],
      "operator": "SAMPLE",
      "sequence": 6,
      "type": "REAL"
    },
    {
      "in_nodes": [
        3
      ],
      "operator": "SAMPLE",
      "sequence": 7,
      "type": "REAL"
    }
  ]
})";

TEST(json_test, test_from_string) {
  folly::dynamic parsed = folly::parseJson(raw_json);
  auto graph = beanmachine::minibmg::json_to_graph(parsed);
  std::string s =
      folly::toPrettyJson(beanmachine::minibmg::graph_to_json(graph));
  ASSERT_EQ(raw_json, s);
}

std::string raw_json_without_types = R"({
  "comment": "created by graph_to_json",
  "queries": [ 2 ],
  "observations": [
    { "node": 4, "value": 1.0 },
    { "node": 5, "value": 1.0 },
    { "node": 6, "value": 1.0 },
    { "node": 7, "value": 0.0 }
  ],
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
      "sequence": 1,
    },
    {
      "in_nodes": [
        1
      ],
      "operator": "SAMPLE",
      "sequence": 2,
    },
    {
      "in_nodes": [
        2
      ],
      "operator": "DISTRIBUTION_BERNOULLI",
      "sequence": 3,
    },
    {
      "in_nodes": [
        3
      ],
      "operator": "SAMPLE",
      "sequence": 4,
    },
    {
      "in_nodes": [
        3
      ],
      "operator": "SAMPLE",
      "sequence": 5,
    },
    {
      "in_nodes": [
        3
      ],
      "operator": "SAMPLE",
      "sequence": 6,
    },
    {
      "in_nodes": [
        3
      ],
      "operator": "SAMPLE",
      "sequence": 7,
    },
  ]
})";

TEST(json_test, test_from_string_without_types) {
  folly::dynamic parsed = folly::parseJson(raw_json_without_types);
  auto graph = json_to_graph(parsed);
  std::string s = folly::toPrettyJson(graph_to_json(graph));
  ASSERT_EQ(raw_json, s);
}
