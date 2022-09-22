/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include "beanmachine/minibmg/ad/number.h"
#include "beanmachine/minibmg/ad/real.h"

using namespace ::testing;
using namespace ::beanmachine::minibmg;

#define EXPECT_RSAME(a, b) EXPECT_DOUBLE_EQ((a).as_double(), b)

template <class T>
requires Number<T>
inline T f1(const T& a, const T& b) {
  return 1.45 * (a * a * a) + 2.43 * (a * a * b) + 3.58 * (a * b * b) +
      4.73 * (b * b * b) + 5.73 * (a * a) + 6.34 * (a * b) + 7.12 * (b * b) +
      8.11 * a + 9.51 * b + 10.73;
}

inline double f2(double a, double b) {
  return 1.45 * (a * a * a) + 2.43 * (a * a * b) + 3.58 * (a * b * b) +
      4.73 * (b * b * b) + 5.73 * (a * a) + 6.34 * (a * b) + 7.12 * (b * b) +
      8.11 * a + 9.51 * b + 10.73;
}

// Set this to true (temporarily) to benchmark the performance of Real vs
// double.
const bool should_benchmark_real_performance = false;

TEST(realperf_test, performance) {
  const long n = should_benchmark_real_performance ? 5'000'000'000 : 120;

  // Test perf with Real
  auto start = std::chrono::high_resolution_clock::now();
  Real ra = 0.554;
  Real rb = 0.128;
  Real rsum = 0;
  for (long i = 0; i < n; i++) {
    rsum = rsum + f1<Real>(ra, rb);
  }
  auto finish = std::chrono::high_resolution_clock::now();
  auto real_time_in_microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(finish - start)
          .count();

  // Test perf with double
  start = std::chrono::high_resolution_clock::now();
  double a = 0.554;
  double b = 0.128;
  double sum = 0;
  for (long i = 0; i < n; i++) {
    sum = sum + f2(a, b);
  }
  finish = std::chrono::high_resolution_clock::now();
  auto double_time_in_microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(finish - start)
          .count();

  EXPECT_RSAME(rsum, sum);
  if (should_benchmark_real_performance) {
    // This test always fails, but prints out the relative performance data.
    EXPECT_TRUE(false) << fmt::format(
                              "Real: {0} s; double: {1} s\n",
                              real_time_in_microseconds / 1E6,
                              double_time_in_microseconds / 1E6)
                       << fmt::format(
                              "Real perf degradation {0} %\n",
                              100.0 *
                                  (real_time_in_microseconds -
                                   double_time_in_microseconds) /
                                  double_time_in_microseconds);
  }
}
