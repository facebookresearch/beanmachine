/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <stdexcept>
#include "beanmachine/minibmg/ad/traced.h"
#include "beanmachine/minibmg/rewriters/dedag.h"

using namespace ::testing;
using namespace beanmachine::minibmg;

TEST(dedag_test, depth) {
  auto x = Traced::variable("x", 1);
  auto deep = x + x + x + x + x + x + x + x + x + x;
  ASSERT_THROW(dedag(deep, 1);, std::invalid_argument);
  auto dedagged = dedag(deep, /* max_depth = */ 2);
  ASSERT_EQ(dedagged.prelude.size(), 9);
}

TEST(dedag_test, simple) {
  auto x = Traced::variable("x", 1);
  auto d1 = x + x;
  auto d2 = d1 + d1;
  auto d3 = d2 + d2;
  auto final = d3 + d3;
  auto dedagged = dedag(final, /* max_depth = */ 20);
  ASSERT_EQ(dedagged.prelude.size(), 3);
}
