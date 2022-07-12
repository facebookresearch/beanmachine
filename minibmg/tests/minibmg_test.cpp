/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "beanmachine/minibmg/minibmg.h"

using namespace ::testing;
using namespace beanmachine::minibmg;

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

TEST(test_minibmg, operator_from_name) {
  ASSERT_EQ(operator_from_name("CONSTANT"), Operator::CONSTANT);
  ASSERT_EQ(operator_from_name("ADD"), Operator::ADD);
  ASSERT_EQ(operator_from_name("MULTIPLY"), Operator::MULTIPLY);
  ASSERT_EQ(
      operator_from_name("DISTRIBUTION_NORMAL"), Operator::DISTRIBUTION_NORMAL);
  ASSERT_EQ(
      operator_from_name("DISTRIBUTION_BETA"), Operator::DISTRIBUTION_BETA);
  ASSERT_EQ(
      operator_from_name("DISTRIBUTION_BERNOULLI"),
      Operator::DISTRIBUTION_BERNOULLI);
  ASSERT_EQ(operator_from_name("SAMPLE"), Operator::SAMPLE);
  ASSERT_EQ(operator_from_name("OBSERVE"), Operator::OBSERVE);
  ASSERT_EQ(operator_from_name("QUERY"), Operator::QUERY);

  ASSERT_EQ(operator_from_name("GIBBERISH"), Operator::NO_OPERATOR);
}

TEST(test_minibmg, operator_to_string) {
  ASSERT_EQ(to_string(Operator::CONSTANT), "CONSTANT");
  ASSERT_EQ(to_string(Operator::ADD), "ADD");
  ASSERT_EQ(to_string(Operator::MULTIPLY), "MULTIPLY");
  ASSERT_EQ(to_string(Operator::DISTRIBUTION_NORMAL), "DISTRIBUTION_NORMAL");
  ASSERT_EQ(to_string(Operator::DISTRIBUTION_BETA), "DISTRIBUTION_BETA");
  ASSERT_EQ(
      to_string(Operator::DISTRIBUTION_BERNOULLI), "DISTRIBUTION_BERNOULLI");
  ASSERT_EQ(to_string(Operator::SAMPLE), "SAMPLE");
  ASSERT_EQ(to_string(Operator::OBSERVE), "OBSERVE");
  ASSERT_EQ(to_string(Operator::QUERY), "QUERY");

  // with runtime checks enabled, the following would crash at the cast.
  // ASSERT_EQ(to_string((Operator)10000), "NO_OPERATOR");
}

TEST(test_minibmg, type_from_name) {
  ASSERT_EQ(type_from_name("NONE"), Type::NONE);
  ASSERT_EQ(type_from_name("DISTRIBUTION"), Type::DISTRIBUTION);
  ASSERT_EQ(type_from_name("REAL"), Type::REAL);

  ASSERT_EQ(type_from_name("GIBBERISH"), Type::NONE);
}

TEST(test_minibmg, type_to_string) {
  ASSERT_EQ(to_string(Type::NONE), "NONE");
  ASSERT_EQ(to_string(Type::DISTRIBUTION), "DISTRIBUTION");
  ASSERT_EQ(to_string(Type::REAL), "REAL");

  // with runtime checks enabled, the following would crash at the cast.
  // ASSERT_EQ(to_string((Type)10000), "NONE");
}
