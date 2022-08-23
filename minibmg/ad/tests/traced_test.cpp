/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include "beanmachine/minibmg/ad/num2.h"
#include "beanmachine/minibmg/ad/traced.h"

using namespace ::testing;
using namespace ::beanmachine::minibmg;

TEST(traced_test, simple1) {
  Traced x = Traced::variable("x", 0);
  auto r = 100 * x.pow(3) + 10 * x.pow(2) + 1 * x - 5;
  ASSERT_EQ("100 * x.pow(3) + 10 * x.pow(2) + x - 5", r.to_string());
}

TEST(traced_test, simple2) {
  Traced x = Traced::variable("x", 0);
  auto r = x.pow(-x);
  ASSERT_EQ("x.pow(-x)", r.to_string());
}

TEST(traced_test, simple3) {
  Traced x = Traced::variable("x", 0);
  auto r = (x + 1) / (x + 2);
  ASSERT_EQ("(x + 1) / (x + 2)", r.to_string());
}

TEST(traced_test, simple4) {
  Traced x = Traced::variable("x", 0);
  auto r = x.exp() + x.log() + x.atan() + x.lgamma() + x.polygamma(5);
  ASSERT_EQ(
      "x.exp() + x.log() + x.atan() + x.lgamma() + x.polygamma(5)",
      r.to_string());
}

TEST(traced_test, simple5) {
  Traced x = Traced::variable("x", 0);
  Traced y = Traced::variable("y", 1);
  Traced z = Traced::variable("z", 2);
  Traced w = Traced::variable("w", 3);
  auto r = x.if_equal(y, z, w);
  ASSERT_EQ("x.if_equal(y, z, w)", r.to_string());
}

TEST(traced_test, simple6) {
  Traced x = Traced::variable("x", 0);
  Traced y = Traced::variable("y", 1);
  Traced z = Traced::variable("z", 2);
  Traced w = Traced::variable("w", 3);
  auto r = x.if_less(y, z, w);
  ASSERT_EQ("x.if_less(y, z, w)", r.to_string());
}

// Show what happens when the computation graph is a dag instead of a tree.
// In that case we unfortunately repeat code in the to_string form.
// We will fix that later (T128357330).
TEST(traced_test, dag) {
  Traced x = Traced::variable("x", 0);
  auto t1 = x + x;
  auto t2 = t1 + t1;
  auto t3 = t2 + t2;
  // As a small consolation, at least it is minimally parenthesized to preserve
  // order of operations.
  ASSERT_EQ("x + x + (x + x) + (x + x + (x + x))", t3.to_string());
}

TEST(traced_test, derivative1) {
  Traced tx = Traced::variable("x", 0);
  Num2<Traced> x{tx, 1};
  auto r = x.pow(2) + 10 * x + 100;
  auto rp = r.derivative1();
  ASSERT_EQ("2 * x + 10", rp.to_string());
}
