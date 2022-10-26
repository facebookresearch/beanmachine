/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <list>
#include "beanmachine/minibmg/graph2.h"
#include "beanmachine/minibmg/graph2_factory.h"
#include "beanmachine/minibmg/graph_properties/out_nodes2.h"

using namespace ::testing;
using namespace ::beanmachine::minibmg;

TEST(out_nodes_test, simple) {
  Graph2::Factory gf;
  auto k12 = gf.constant(1.2);
  auto k34 = gf.constant(3.4);
  auto plus = gf.add(k12, k34);
  auto k56 = gf.constant(5.6);
  auto beta = gf.beta(plus, k56);
  auto sample = gf.sample(beta);
  gf.observe(sample, 7.8);
  /* auto query_ = */ gf.query(sample);
  Graph2 g = gf.build();

  Node2p k12n = gf[k12];
  Node2p k34n = gf[k34];
  Node2p plusn = gf[plus];
  Node2p k56n = gf[k56];
  Node2p betan = gf[beta];
  Node2p samplen = gf[sample];

  ASSERT_EQ(out_nodes(g, k12n), std::list{plusn});
  ASSERT_EQ(out_nodes(g, k34n), (std::list{plusn}));
  ASSERT_EQ(out_nodes(g, plusn), std::list<Node2p>{betan});
  ASSERT_EQ(out_nodes(g, k56n), std::list{betan});
  ASSERT_EQ(out_nodes(g, betan), (std::list{samplen}));
  ASSERT_EQ(out_nodes(g, samplen), std::list<Node2p>{});

  ASSERT_EQ(g.queries, std::vector{samplen});
  std::list<std::pair<Node2p, double>> expected_observations;
  expected_observations.push_back(std::pair{samplen, 7.8});
  ASSERT_EQ(g.observations, expected_observations);
}

TEST(out_nodes_test, not_found2) {
  Graph2::Factory gf;
  Graph2 g = gf.build();
  Node2p n = std::make_shared<ScalarConstantNode2>(0);
  ASSERT_THROW(out_nodes(g, n), std::invalid_argument);
}
