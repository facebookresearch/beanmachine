/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "beanmachine/graph/marginalization/copy_node.h"

using namespace beanmachine;
using namespace graph;

TEST(testmarginal, copynode) {
  Graph g;
  uint zero = g.add_constant(0.0);
  uint one = g.add_constant_pos_real(1.0);
  uint normal = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {zero, one});
  uint normal_sample = g.add_operator(OperatorType::SAMPLE, {normal});
  g.observe(normal_sample, 0.5);

  Node* normal_node = g.get_node(normal_sample);
  CopyNode copy_node = CopyNode(normal_node);
  EXPECT_EQ(normal_node->value, copy_node.value);
}
