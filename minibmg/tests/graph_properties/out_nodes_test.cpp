/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <vector>
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/graph_factory.h"
#include "beanmachine/minibmg/graph_properties/out_nodes.h"

using namespace ::testing;
using namespace ::beanmachine::minibmg;

TEST(out_nodes_test, simple) {
  Graph::Factory gf;
  auto k12 = gf.constant(1.2);
  auto k34 = gf.constant(3.4);
  auto plus = gf.add(k12, k34);
  auto k56 = gf.constant(5.6);
  auto beta = gf.beta(plus, k56);
  auto sample = gf.sample(beta);
  gf.observe(sample, 7.8);
  /* auto query_ = */ gf.query(sample);
  Graph g = gf.build();

  Nodep k12n = gf[k12];
  Nodep k34n = gf[k34];
  Nodep plusn = gf[plus];
  Nodep k56n = gf[k56];
  Nodep betan = gf[beta];
  Nodep samplen = gf[sample];

  ASSERT_EQ(out_nodes(g, k12n), std::vector{plusn});
  ASSERT_EQ(out_nodes(g, k34n), (std::vector{plusn}));
  ASSERT_EQ(out_nodes(g, plusn), std::vector<Nodep>{betan});
  ASSERT_EQ(out_nodes(g, k56n), std::vector{betan});
  ASSERT_EQ(out_nodes(g, betan), (std::vector{samplen}));
  ASSERT_EQ(out_nodes(g, samplen), std::vector<Nodep>{});

  ASSERT_EQ(g.queries, std::vector{samplen});
  std::vector<std::pair<Nodep, double>> expected_observations;
  expected_observations.push_back(std::pair{samplen, 7.8});
  ASSERT_EQ(g.observations, expected_observations);
}

TEST(out_nodes_test, not_found2) {
  Graph::Factory gf;
  Graph g = gf.build();
  Nodep n = std::make_shared<ScalarConstantNode>(0);
  ASSERT_THROW(out_nodes(g, n), std::invalid_argument);
}
