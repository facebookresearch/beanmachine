/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "beanmachine/minibmg/minibmg.h"

using namespace ::testing;

namespace beanmachine::minibmg {

TEST(test_minibmg, basic_building) {
  Graph::Factory gf;
  auto k12 = gf.add_constant(1.2);
  ASSERT_EQ(k12, 0);
  auto k34 = gf.add_constant(3.4);
  ASSERT_EQ(k34, 1);
  auto plus = gf.add_operator(Operator::ADD, {k12, k34});
  ASSERT_EQ(plus, 2);
  auto k56 = gf.add_constant(5.6);
  ASSERT_EQ(k56, 3);
  auto beta = gf.add_operator(Operator::DISTRIBUTION_BETA, {k34, k56});
  ASSERT_EQ(beta, 4);
  auto sample = gf.add_operator(Operator::SAMPLE, {beta});
  ASSERT_EQ(sample, 5);
  auto k78 = gf.add_constant(7.8);
  ASSERT_EQ(k78, 6);
  auto observe = gf.add_operator(Operator::OBSERVE, {beta, k78});
  ASSERT_EQ(observe, 7);
  auto query = gf.add_query(beta);
  ASSERT_EQ(query, 0); // we get the query number back from add_query
  Graph g = gf.build();
  ASSERT_EQ(g.nodes.size(), 9);
}

} // namespace beanmachine::minibmg
