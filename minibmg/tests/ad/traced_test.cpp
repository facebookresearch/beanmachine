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

TEST(traced_test, simple0) {
  Traced x = Traced::variable("x", 0);
  Traced r = 100 + x;
  ASSERT_EQ("100 + x", to_string(r));
}

TEST(traced_test, simple1) {
  Traced x = Traced::variable("x", 0);
  auto r = 100 * pow(x, 3) + 10 * pow(x, 2) + 1 * x - 5;
  ASSERT_EQ("100 * pow(x, 3) + 10 * pow(x, 2) + 1 * x - 5", to_string(r));
}

TEST(traced_test, simple2) {
  Traced x = Traced::variable("x", 0);
  auto r = pow(x, -x);
  ASSERT_EQ("pow(x, -x)", to_string(r));
}

TEST(traced_test, simple3) {
  Traced x = Traced::variable("x", 0);
  auto r = (x + 1) / (x + 2);
  ASSERT_EQ("(x + 1) / (x + 2)", to_string(r));
}

TEST(traced_test, simple4) {
  Traced x = Traced::variable("x", 0);
  auto r = exp(x) + log(x) + log1p(x) + atan(x) + lgamma(x) + polygamma(5, x);
  ASSERT_EQ(
      "exp(x) + log(x) + log1p(x) + atan(x) + lgamma(x) + polygamma(5, x)",
      to_string(r));
}

TEST(traced_test, simple5) {
  Traced x = Traced::variable("x", 0);
  Traced y = Traced::variable("y", 1);
  Traced z = Traced::variable("z", 2);
  Traced w = Traced::variable("w", 3);
  auto r = if_equal(x, y, z, w);
  ASSERT_EQ("if_equal(x, y, z, w)", to_string(r));
}

TEST(traced_test, simple6) {
  Traced x = Traced::variable("x", 0);
  Traced y = Traced::variable("y", 1);
  Traced z = Traced::variable("z", 2);
  Traced w = Traced::variable("w", 3);
  auto r = if_less(x, y, z, w);
  ASSERT_EQ("if_less(x, y, z, w)", to_string(r));
}

// Show what happens when the computation graph is a dag instead of a tree.
TEST(traced_test, dag) {
  Traced x = Traced::variable("x", 0);
  auto t1 = x + x;
  auto t2 = t1 + t1;
  auto t3 = t2 + t2;
  std::string expected_result =
      R":(auto temp_1 = x + x;
auto temp_2 = temp_1 + temp_1;
temp_2 + temp_2):";
  ASSERT_EQ(expected_result, to_string(t3));
}

TEST(traced_test, derivative1) {
  Traced tx = Traced::variable("x", 0);
  Num2<Traced> x{tx, 1};
  auto r = pow(x, 2) + 10 * x + 100;
  auto rp = r.derivative1;
  auto rpopt = opt(rp);
  ASSERT_EQ("2 * x + 10", to_string(rpopt));
}
