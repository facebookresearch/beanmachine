/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <random>
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/minibmg.h"

using namespace ::testing;
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
    EXPECT_RSAME((r1 + 2).pow(r2), std::pow(d1 + 2, d2));
    EXPECT_RSAME(r1.exp(), std::exp(d1));
    EXPECT_RSAME((r1 + 2).log(), std::log(d1 + 2));
    EXPECT_RSAME(r1.atan(), std::atan(d1));
    EXPECT_RSAME(r1.polygamma(0), boost::math::polygamma(0, d1));
    EXPECT_RSAME(r1.polygamma(1), boost::math::polygamma(1, d1));
    EXPECT_RSAME(r1.polygamma(2), boost::math::polygamma(2, d1));
    EXPECT_RSAME(r1.if_equal(r1, r2, r3), d2);
    EXPECT_RSAME(r1.if_equal(r2, r3, r4), d4);
    EXPECT_RSAME(r1.if_less(r2, r3, r4), (d1 < d2) ? d3 : d4);
    EXPECT_RSAME(r2.if_less(r1, r3, r4), (d2 < d1) ? d3 : d4);
  }
}

TEST(real_test, definite) {
  EXPECT_TRUE(Real(0).is_definitely_zero());
  EXPECT_FALSE(Real(0.001).is_definitely_zero());
  EXPECT_TRUE(Real(1).is_definitely_one());
  EXPECT_FALSE(Real(1.001).is_definitely_one());
}
