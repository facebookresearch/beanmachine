/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "beanmachine/minibmg/minibmg.h"

using namespace ::testing;

std::string raw_json = R"({
  "comment": "created by graph_to_json",
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
      "operator": "CONSTANT",
      "sequence": 4,
      "type": "REAL",
      "value": 0
    },
    {
      "operator": "CONSTANT",
      "sequence": 5,
      "type": "REAL",
      "value": 1
    },
    {
      "in_nodes": [
        3,
        5
      ],
      "operator": "OBSERVE",
      "sequence": 6,
      "type": "NONE"
    },
    {
      "in_nodes": [
        3,
        5
      ],
      "operator": "OBSERVE",
      "sequence": 7,
      "type": "NONE"
    },
    {
      "in_nodes": [
        3,
        5
      ],
      "operator": "OBSERVE",
      "sequence": 8,
      "type": "NONE"
    },
    {
      "in_nodes": [
        3,
        4
      ],
      "operator": "OBSERVE",
      "sequence": 9,
      "type": "NONE"
    },
    {
      "in_node": 2,
      "operator": "QUERY",
      "query_index": 0,
      "sequence": 10,
      "type": "NONE"
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
  "nodes": [
    {
      "sequence": 0,
      "operator": "CONSTANT",
      "value": 2
    },
    {
      "sequence": 1,
      "operator": "DISTRIBUTION_BETA",
      "in_nodes": [
        0,
        0
      ]
    },
    {
      "sequence": 2,
      "operator": "SAMPLE",
      "in_nodes": [
        1
      ]
    },
    {
      "sequence": 3,
      "operator": "DISTRIBUTION_BERNOULLI",
      "in_nodes": [
        2
      ]
    },
    {
      "sequence": 4,
      "operator": "CONSTANT",
      "value": 0
    },
    {
      "sequence": 5,
      "operator": "CONSTANT",
      "value": 1
    },
    {
      "sequence": 6,
      "operator": "OBSERVE",
      "in_nodes": [
        3,
        5
      ]
    },
    {
      "sequence": 7,
      "operator": "OBSERVE",
      "in_nodes": [
        3,
        5
      ]
    },
    {
      "sequence": 8,
      "operator": "OBSERVE",
      "in_nodes": [
        3,
        5
      ]
    },
    {
      "sequence": 9,
      "operator": "OBSERVE",
      "in_nodes": [
        3,
        4
      ]
    },
    {
      "sequence": 10,
      "operator": "QUERY",
      "in_node": 2,
      "query_index": 0
    }
  ]
})";

TEST(json_test, test_from_string_without_types) {
  folly::dynamic parsed = folly::parseJson(raw_json_without_types);
  auto graph = beanmachine::minibmg::json_to_graph(parsed);
  std::string s =
      folly::toPrettyJson(beanmachine::minibmg::graph_to_json(graph));
  ASSERT_EQ(raw_json, s);
}
