/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include "beanmachine/minibmg/fluent_factory.h"
#include "beanmachine/minibmg/graph.h"

using namespace ::testing;
using namespace beanmachine::minibmg;

namespace fluent_factory_test {

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
      "operator": "CONSTANT",
      "sequence": 1,
      "type": "REAL",
      "value": 2
    },
    {
      "in_nodes": [
        0,
        1
      ],
      "operator": "DISTRIBUTION_BETA",
      "sequence": 2,
      "type": "DISTRIBUTION"
    },
    {
      "in_nodes": [
        2
      ],
      "operator": "SAMPLE",
      "sequence": 3,
      "type": "REAL"
    },
    {
      "in_nodes": [
        3
      ],
      "operator": "DISTRIBUTION_BERNOULLI",
      "sequence": 4,
      "type": "DISTRIBUTION"
    },
    {
      "in_nodes": [
        4
      ],
      "operator": "SAMPLE",
      "sequence": 5,
      "type": "REAL"
    },
    {
      "in_nodes": [
        4
      ],
      "operator": "SAMPLE",
      "sequence": 6,
      "type": "REAL"
    },
    {
      "in_nodes": [
        4
      ],
      "operator": "SAMPLE",
      "sequence": 7,
      "type": "REAL"
    },
    {
      "in_nodes": [
        4
      ],
      "operator": "SAMPLE",
      "sequence": 8,
      "type": "REAL"
    },
    {
      "in_nodes": [
        4
      ],
      "operator": "SAMPLE",
      "sequence": 9,
      "type": "REAL"
    }
  ],
  "observations": [
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
      "value": 1
    },
    {
      "node": 8,
      "value": 0
    },
    {
      "node": 9,
      "value": 0
    }
  ],
  "queries": [
    3
  ]
})";

TEST(fluent_factory_test, simple_test) {
  Graph::FluentFactory fac;
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
  auto json = folly::toPrettyJson(beanmachine::minibmg::graph_to_json(graph));
  ASSERT_EQ(raw_json, json);
}

} // namespace fluent_factory_test
