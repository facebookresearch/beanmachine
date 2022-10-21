/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <stdexcept>

#include "beanmachine/minibmg/dedup.h"
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/graph_factory.h"
#include "beanmachine/minibmg/node.h"

using namespace ::testing;
using namespace beanmachine::minibmg;

#define ASSERT_ID(node, num) ASSERT_EQ(node->_value(), num)

TEST(test_minibmg, basic_building_1) {
  Graph::Factory gf;
  auto k12 = gf.constant(1.2);
  ASSERT_ID(k12, 0);
  auto k34 = gf.constant(3.4);
  ASSERT_ID(k34, 1);
  auto plus = gf.add(k12, k34);
  ASSERT_ID(plus, 2);
  auto k56 = gf.constant(5.6);
  ASSERT_ID(k56, 3);
  auto beta = gf.beta(plus, k56);
  ASSERT_ID(beta, 4);
  auto sample = gf.sample(beta);
  ASSERT_ID(sample, 5);
  gf.observe(sample, 7.8);
  auto query = gf.query(sample);
  ASSERT_EQ(query, 0);
  Graph g = gf.build();
  ASSERT_EQ(g.size(), 6);
}

TEST(test_minibmg, dead_code_dropped) {
  Graph::Factory gf;
  auto k12 = gf.constant(1.2);
  ASSERT_ID(k12, 0);
  auto k34 = gf.constant(3.4);
  ASSERT_ID(k34, 1);
  auto plus = gf.add(k12, k34);
  ASSERT_ID(plus, 2);
  auto k56 = gf.constant(5.6);
  ASSERT_ID(k56, 3);
  auto beta = gf.beta(k34, k56);
  ASSERT_ID(beta, 4);
  auto sample = gf.sample(beta);
  ASSERT_ID(sample, 5);
  gf.observe(sample, 7.8);
  auto query = gf.query(sample);
  ASSERT_EQ(query, 0);
  Graph g = gf.build();
  ASSERT_EQ(g.size(), 4); // k34 and plus are dead code.
}

TEST(test_minibmg, duplicate_build) {
  Graph::Factory gf;
  Graph g = gf.build();
  ASSERT_THROW(gf.constant(1.2);, std::invalid_argument);
  ASSERT_THROW(gf.build();, std::invalid_argument);
}

// tests the Dedupable concept
TEST(test_minibmg, dedupable_concept) {
  ASSERT_TRUE(Dedupable<std::vector<Nodep>>);
  ASSERT_TRUE(Dedupable<std::vector<std::vector<Nodep>>>);
  ASSERT_FALSE(Dedupable<std::vector<int>>);
  ASSERT_FALSE(Dedupable<std::vector<std::vector<int>>>);
}
