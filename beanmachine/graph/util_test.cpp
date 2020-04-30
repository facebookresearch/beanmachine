// Copyright (c) Facebook, Inc. and its affiliates.
#include <chrono>
#include <random>

#include <gtest/gtest.h>

#include "beanmachine/graph/util.h"

using namespace beanmachine;

#define LOG_90 ((double)4.49980967033)
#define LOG_10 ((double)2.30258509299)
#define LOG_PT_01 ((double)-4.60517018599)

TEST(testutil, sample_logodds) {
  // obtain a seed from the system clock:
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 gen(seed);
  // check the basic functionality with large weights
  int cnt_pos = 0;
  for (int i = 0; i < 10000; i++) {
    if (util::sample_logodds(gen, LOG_90 - LOG_10)) {
      cnt_pos += 1;
    }
  }
  // true odds is 0.9 and the odds of either of these being violated
  // is less than 1 in million
  EXPECT_GT(cnt_pos, 8800);
  EXPECT_LT(cnt_pos, 9200);
  cnt_pos = 0;
  for (int i = 0; i < 10000; i++) {
    if (util::sample_logodds(gen, LOG_10 - LOG_90)) {
      cnt_pos += 1;
    }
  }
  // true odds is 0.1 and the odds of either of these being violated
  // is less than 1 in million
  EXPECT_GT(cnt_pos, 800);
  EXPECT_LT(cnt_pos, 1200);
  // now check with some disparity in weights
  cnt_pos = 0;
  for (int i = 0; i < 100000; i++) {
    if (util::sample_logodds(gen, LOG_PT_01 - LOG_90)) {
      cnt_pos += 1;
    }
  }
  // true odds is 0.00011109876 so the odds of failing this test is less than
  // two in 100K
  EXPECT_GT(cnt_pos, 0);
  // odds of failing the following is less than 1 in million
  EXPECT_LT(cnt_pos, 30);
}

TEST(testutil, Phi) {
  // We will use the fact that Phi(-1.96) = 0.025 = 1 - Phi(1.96)
  // See https://en.wikipedia.org/wiki/Probit
  EXPECT_NEAR(util::Phi(-1.96), 0.025, 0.001);
  EXPECT_NEAR(util::Phi_approx(-1.96), 0.025, 0.01);
  EXPECT_NEAR(util::Phi(1.96), 1 - 0.025, 0.001);
  EXPECT_NEAR(util::Phi_approx(1.96), 1 - 0.025, 0.01);
  EXPECT_NEAR(util::Phi_approx_inv(0.025), -1.96, 0.2);
  EXPECT_NEAR(util::Phi_approx_inv(1 - 0.025), 1.96, 0.2);
}

TEST(testutil, log_sum_exp) {
  std::vector<double> vals = {-3, -5, -7};
  EXPECT_NEAR(util::log_sum_exp(vals), -2.8571, 0.001);
}
