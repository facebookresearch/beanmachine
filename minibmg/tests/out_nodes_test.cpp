/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <list>
#include <unordered_set>
#include "beanmachine/minibmg/minibmg.h"
#include "beanmachine/minibmg/out_nodes.h"

using namespace ::testing;
using namespace ::beanmachine::minibmg;

TEST(out_nodes_test, simple) {
  Graph::Factory gf;
  auto k12 = gf.add_constant(1.2);
  auto k34 = gf.add_constant(3.4);
  auto plus = gf.add_operator(Operator::ADD, {k12, k34});
  auto k56 = gf.add_constant(5.6);
  auto beta = gf.add_operator(Operator::DISTRIBUTION_BETA, {k34, k56});
  auto sample = gf.add_operator(Operator::SAMPLE, {beta});
  auto k78 = gf.add_constant(7.8);
  auto observe = gf.add_operator(Operator::OBSERVE, {beta, k78});
  NodeId query;
  /* auto query_ = */ gf.add_query(beta, query);

  Nodep k12n = gf[k12];
  Nodep k34n = gf[k34];
  Nodep plusn = gf[plus];
  Nodep k56n = gf[k56];
  Nodep betan = gf[beta];
  Nodep samplen = gf[sample];
  Nodep k78n = gf[k78];
  Nodep observen = gf[observe];
  Nodep queryn = gf[query];
  Graph g = gf.build();

  ASSERT_EQ(out_nodes(g, k12n), std::list{plusn});
  ASSERT_EQ(out_nodes(g, k34n), (std::list{plusn, betan}));
  ASSERT_EQ(out_nodes(g, plusn), std::list<Nodep>{});
  ASSERT_EQ(out_nodes(g, k56n), std::list{betan});
  ASSERT_EQ(out_nodes(g, betan), (std::list{samplen, observen, queryn}));
  ASSERT_EQ(out_nodes(g, samplen), std::list<Nodep>{});
  ASSERT_EQ(out_nodes(g, k78n), std::list{observen});
  ASSERT_EQ(out_nodes(g, observen), std::list<Nodep>{});
  ASSERT_EQ(out_nodes(g, queryn), std::list<Nodep>{});
}

TEST(out_nodes_test, not_found2) {
  Graph::Factory gf;
  Graph g = gf.build();
  Nodep n = std::make_shared<const ConstantNode>(0);
  ASSERT_THROW(out_nodes(g, n), std::invalid_argument);
}
