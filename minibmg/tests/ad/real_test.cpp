/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <random>
#include "beanmachine/minibmg/ad/real.h"

using namespace ::beanmachine::minibmg;

#define EXPECT_RSAME(a, b) EXPECT_DOUBLE_EQ((a).as_double(), b)

TEST(real_test, initialization) {
  std::mt19937 g;
  std::uniform_real_distribution<double> unif(-2.0, 2.0);

  for (int i = 0; i < 5; i++) {
    double d = unif(g);
    Real r = d;
    EXPECT_RSAME(r, d);
    Real r2 = r;
    EXPECT_RSAME(r2, d);
    Real r3{d};
    EXPECT_RSAME(r3, d);
    Real r4{r};
    EXPECT_RSAME(r4, d);
  }
}

TEST(real_test, computations) {
  std::mt19937 g;
  std::uniform_real_distribution<double> unif(-2.0, 2.0);

  for (int i = 0; i < 5; i++) {
    double d1 = unif(g), d2 = unif(g), d3 = unif(g), d4 = unif(g);
    Real r1 = d1, r2 = d2, r3 = d3, r4 = d4;

    EXPECT_RSAME(r1 + r2, d1 + d2);
    EXPECT_RSAME(r1 - r2, d1 - d2);
    EXPECT_RSAME(-r1, -d1);
    EXPECT_RSAME(r1 * r2, d1 * d2);
    EXPECT_RSAME(r1 / r2, d1 / d2);
    EXPECT_RSAME(pow(r1 + 2, r2), std::pow(d1 + 2, d2));
    EXPECT_RSAME(exp(r1), std::exp(d1));
    EXPECT_RSAME(log(r1 + 2), std::log(d1 + 2));
    EXPECT_RSAME(atan(r1), std::atan(d1));
    EXPECT_RSAME(polygamma(0, r1), boost::math::polygamma(0, d1));
    EXPECT_RSAME(polygamma(1, r1), boost::math::polygamma(1, d1));
    EXPECT_RSAME(polygamma(2, r1), boost::math::polygamma(2, d1));
    EXPECT_RSAME(if_equal(r1, r1, r2, r3), d2);
    EXPECT_RSAME(if_equal(r1, r2, r3, r4), d4);
    EXPECT_RSAME(if_less(r1, r2, r3, r4), (d1 < d2) ? d3 : d4);
    EXPECT_RSAME(if_less(r2, r1, r3, r4), (d2 < d1) ? d3 : d4);
  }
}

TEST(real_test, definite) {
  EXPECT_TRUE(is_constant(Real(0), 0));
  EXPECT_FALSE(is_constant(Real(0.001), 0));
  EXPECT_TRUE(is_constant(Real(1), 1));
  EXPECT_FALSE(is_constant(Real(1.001), 1));
}

// Tests the "number" concept
TEST(real_test, concept) {
  EXPECT_TRUE(Number<Real>);
  // double isn't a number because you cannot use some operations without
  // importing an appropriate namespace.
  EXPECT_FALSE(Number<double>);
}
