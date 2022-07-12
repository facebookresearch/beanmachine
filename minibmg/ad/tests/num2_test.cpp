/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <cmath>
#include "beanmachine/minibmg/ad/num2.h"
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/ad/tests/test_utils.h"

using namespace ::testing;
using namespace beanmachine::minibmg;

using Dual = Num2<Real>;

TEST(num2_test, convert) {
  Dual d0{0};
  EXPECT_TRUE(d0.is_definitely_zero());
  EXPECT_FALSE(d0.is_definitely_one());
  Dual d1(1);
  EXPECT_FALSE(d1.is_definitely_zero());
  EXPECT_TRUE(d1.is_definitely_one());
  Dual d2(2);
  EXPECT_FALSE(d2.is_definitely_zero());
  EXPECT_FALSE(d2.is_definitely_one());
}

TEST(num2_test, grad1) {
  Dual d0{0, 1};
  EXPECT_FALSE(d0.is_definitely_zero());
  EXPECT_FALSE(d0.is_definitely_one());
  Dual d1(1, 1);
  EXPECT_FALSE(d1.is_definitely_zero());
  EXPECT_FALSE(d1.is_definitely_one());
  Dual d2(2, 1);
  EXPECT_FALSE(d2.is_definitely_zero());
  EXPECT_FALSE(d2.is_definitely_one());
}

TEST(num2_test, grad2) {
  Dual d0{0, 0};
  EXPECT_TRUE(d0.is_definitely_zero());
  EXPECT_FALSE(d0.is_definitely_one());
  Dual d1(1, 0);
  EXPECT_FALSE(d1.is_definitely_zero());
  EXPECT_TRUE(d1.is_definitely_one());
  Dual d2(2, 0);
  EXPECT_FALSE(d2.is_definitely_zero());
  EXPECT_FALSE(d2.is_definitely_one());
}

TEST(num2_test, add1) {
  Dual a{1.1, 2.2};
  Dual b(3.3, 4.4);
  Dual sum = a + b;
  EXPECT_CLOSE(4.4, sum.primal.as_double());
  EXPECT_CLOSE(6.6, sum.derivative1.as_double());
}

TEST(num2_test, add2) {
  Dual a{1.1, 2.2};
  Dual sum = a + 4;
  EXPECT_CLOSE(5.1, sum.primal.as_double());
  EXPECT_CLOSE(2.2, sum.derivative1.as_double());
}

TEST(num2_test, sub1) {
  Dual a{1.1, 2.2};
  Dual b(3.3, 4.4);
  Dual diff = b - a;
  EXPECT_CLOSE(2.2, diff.primal.as_double());
  EXPECT_CLOSE(2.2, diff.derivative1.as_double());
}

TEST(num2_test, sub2) {
  Dual a{1.1, 2.2};
  Dual diff = a - 1;
  EXPECT_CLOSE(0.1, diff.primal.as_double());
  EXPECT_CLOSE(2.2, diff.derivative1.as_double());
}

TEST(num2_test, negate) {
  Dual a{1.1, 2.2};
  Dual neg = -a;
  EXPECT_CLOSE(-1.1, neg.primal.as_double());
  EXPECT_CLOSE(-2.2, neg.derivative1.as_double());
}

TEST(num2_test, mul1) {
  Dual a{1.1, 2.2};
  Dual b(3.3, 4.4);
  Dual prod = a * b;
  EXPECT_CLOSE(1.1 * 3.3, prod.primal.as_double());
  EXPECT_CLOSE(1.1 * 4.4 + 2.2 * 3.3, prod.derivative1.as_double());
}

TEST(num2_test, div1) {
  Dual a{1.1, 2.2};
  Dual b(3.3, 4.4);
  Dual prod = a / b;
  EXPECT_CLOSE(1.0 / 3, prod.primal.as_double());
  EXPECT_CLOSE(2.0 / 9, prod.derivative1.as_double());
}

TEST(num2_test, pow1) {
  Dual a{1.1, 2.2};
  Dual b(3.3, 4.4);
  Dual pow = a.pow(b);
  double expected_primal = std::pow(a.as_double(), b.as_double());
  // a^b = exp(b log a)
  // d/dx a^b
  // = d/dx exp(b log a)
  // = exp(b log a) * d/dx (b log a)
  // = pow(a, b) * (b d/dx (log a) + d/dx (b) log a)
  // = pow(a, b) * (b a' / a + b' log a)
  double expected_grad1 = expected_primal *
      (b.as_double() * a.derivative1.as_double() / a.as_double() +
       b.derivative1.as_double() * std::log(a.as_double()));
  EXPECT_CLOSE(expected_primal, pow.primal.as_double());
  EXPECT_CLOSE(expected_grad1, pow.derivative1.as_double());
}

TEST(num2_test, exp) {
  Dual a{1.1, 2.2};
  Dual value = a.exp();
  double expected_primal = std::exp(a.as_double());
  double expected_grad1 = a.derivative1.as_double() * expected_primal;
  EXPECT_CLOSE(expected_primal, value.primal.as_double());
  EXPECT_CLOSE(expected_grad1, value.derivative1.as_double());
}

TEST(num2_test, log) {
  Dual a{1.1, 2.3};
  Dual value = a.log();
  double expected_primal = std::log(a.as_double());
  double expected_grad1 = a.derivative1.as_double() / a.as_double();
  EXPECT_CLOSE(expected_primal, value.primal.as_double());
  EXPECT_CLOSE(expected_grad1, value.derivative1.as_double());
}

TEST(num2_test, atan) {
  Dual a{1.1, 2.2};
  Dual value = a.atan();
  double expected_primal = std::atan(a.as_double());
  double expected_grad1 = a.derivative1.as_double() /
      (1 + a.primal.as_double() * a.primal.as_double());
  EXPECT_CLOSE(expected_primal, value.primal.as_double());
  EXPECT_CLOSE(expected_grad1, value.derivative1.as_double());
}

TEST(num2_test, lgamma) {
  Dual a{1.1, 2.2};
  Dual value = a.lgamma();
  double expected_primal = std::lgamma(a.as_double());
  double expected_grad1 =
      a.derivative1.as_double() * boost::math::polygamma(0, a.as_double());
  EXPECT_CLOSE(expected_primal, value.primal.as_double());
  EXPECT_CLOSE(expected_grad1, value.derivative1.as_double());
}

TEST(num2_test, polygamma) {
  Dual a{1.1, 2.2};
  Dual value = a.polygamma(2);
  double expected_primal = boost::math::polygamma(2, a.as_double());
  double expected_grad1 =
      a.derivative1.as_double() * boost::math::polygamma(3, a.as_double());
  EXPECT_CLOSE(expected_primal, value.primal.as_double());
  EXPECT_CLOSE(expected_grad1, value.derivative1.as_double());
}
