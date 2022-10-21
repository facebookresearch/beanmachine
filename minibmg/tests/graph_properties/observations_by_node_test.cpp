/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include "beanmachine/minibmg/fluid_factory.h"
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/graph_properties/observations_by_node.h"

using namespace ::testing;
using namespace beanmachine::minibmg;

TEST(observations_by_node_test, simple) {
  Graph::FluidFactory f;
  auto b = beta(2, 2);
  auto s1 = sample(b);
  auto s2 = sample(b);
  f.observe(s1, 0.5);
  f.observe(s2, 0.4);
  auto g = f.build();
  auto& obs = observations_by_node(g);
  ASSERT_EQ(obs.size(), 2);
  for (auto o : g.observations) {
    ASSERT_EQ(obs.find(o.first)->second, o.second);
  }
}
