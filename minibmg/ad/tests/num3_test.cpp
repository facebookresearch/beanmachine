/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <list>
#include <random>

#include "beanmachine/minibmg/ad/num2.h"
#include "beanmachine/minibmg/ad/num3.h"
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/ad/tests/test_utils.h"

using namespace ::testing;
using namespace beanmachine::minibmg;

using Triune = Num3<Real>;
using Dual = Num2<Real>;
using DualDual = Num2<Dual>;

TEST(num3_test, division_denominator) {
  double k1 = 3.0;
  double k2 = 7.0;
  const Triune x{k1, 1, 0};
  const Triune& y = k2 / x;
  EXPECT_CLOSE(k2 / k1, y.primal.as_double());
  EXPECT_CLOSE(-(k2 / (k1 * k1)), y.derivative1.as_double());
  EXPECT_CLOSE(2 * k2 / (k1 * k1 * k1), y.derivative2.as_double());
}

template <class N>
using BinaryFunction = N (*)(N, N);

// Return a vector of binary functions from (T,T) to T.
template <class T>
requires Number<T> std::vector<BinaryFunction<T>> functions() {
  static std::vector<BinaryFunction<T>> result = {
      // This list of binary functions exercises all of the operations on a
      // Number.
      [](T a, T b) { return a + b; },
      [](T a, T b) { return a - b; },
      [](T a, T b) { return a * b; },
      [](T a, T b) { return a / b; },
      [](T a, T b) { return pow(a, b); },
      [](T a, T b) { return exp(a + b); },
      [](T a, T b) { return log(a + b + 4); },
      [](T a, T b) { return atan(a + b); },
      [](T a, T b) { return lgamma(a + b); },
      [](T a, T b) { return polygamma(0, a + b); },
      [](T a, T b) { return polygamma(1, a + b); },
      [](T a, T b) { return polygamma(2, a + b); },
      [](T a, T b) { return polygamma(3, a + b); },
      [](T a, T b) { return if_equal(a, a, b, a); },
      [](T a, T b) { return if_equal(a, b, b, a); },
      [](T a, T b) { return if_less(a, b, a, b); },
      [](T a, T b) { return if_less(b, a, b, a); },
      [](T a, T b) { return log(exp(a) + exp(b)); },
      [](T a, T b) {
        return 1.2 * a * a * a + 2.6 * a * a * b + 3.14 * a * b * b +
            4.41 * b * b * b;
      },
  };
  return result;
}

// Compare the behavior of nested Dual2 numbers to Dual3.  They should get the
// same result.
TEST(num3_test, compare_to_num2) {
  // Functions on Triune, which is Num3<Real, Real>
  // This type can compute the primal and first and second derivaives.
  auto f1s = functions<Triune>();
  // Functions on DualDual, which is Num2<Num2<Real, Real>, Num2<Real, Real>>
  // This type can also compute the primal and first and second derivaives.
  // Since Num2 is very simple and well tested, we test Num3 by comparing
  // the result of computing various functions.
  auto f2s = functions<DualDual>();

  // Use a random number generator with its default seed.
  std::mt19937 g;
  // We generate several doubles between -2.0 and 2.0.
  std::uniform_real_distribution<double> unif(-2.0, 2.0);

  // we test each function 5 times with different sample values
  for (int k = 0; k < 5; k++) {
    double k1 = unif(g);
    double k2 = unif(g);
    double k3 = unif(g);
    double k4 = unif(g);
    double k5 = unif(g);
    double k6 = unif(g);

    const Triune& t1 = Triune{k1, k2, k3};
    const DualDual& d1 = DualDual{Dual{k1, k2}, Dual{k2, k3}};
    const Triune& t2 = Triune{k4, k5, k6};
    const DualDual& d2 = DualDual{Dual{k4, k5}, Dual{k5, k6}};

    for (int i = 0, n = f1s.size(); i < n; i++) {
      const auto& f1 = f1s[i];
      const auto& f2 = f2s[i];

      const auto& t3 = f1(t1, t2);
      const auto& d3 = f2(d1, d2);

      EXPECT_CLOSE(t3.as_double(), d3.as_double());
      EXPECT_CLOSE(t3.derivative1.as_double(), d3.derivative1.as_double());
      EXPECT_CLOSE(
          t3.derivative1.as_double(), d3.primal.derivative1.as_double());
      EXPECT_CLOSE(
          t3.derivative2.as_double(), d3.derivative1.derivative1.as_double());
    }
  }
}
