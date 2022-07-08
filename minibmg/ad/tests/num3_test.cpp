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
  Triune x{k1, 1, 0};
  Triune y = k2 / x;
  EXPECT_CLOSE(k2 / k1, y.primal.as_double());
  EXPECT_CLOSE(-(k2 / (k1 * k1)), y.derivative1.as_double());
  EXPECT_CLOSE(2 * k2 / (k1 * k1 * k1), y.derivative2.as_double());
}

template <class N>
using BinaryFunction = N (*)(N, N);

// Return a set of binary functions for the number type T.
template <class T>
requires Number<T> std::vector<BinaryFunction<T>> functions() {
  static std::vector<BinaryFunction<T>> result = {
      // A set of functions that exercises all of the operations on a Number.
      [](T a, T b) { return a + b; },
      [](T a, T b) { return a - b; },
      [](T a, T b) { return a * b; },
      [](T a, T b) { return a / b; },
      [](T a, T b) { return a.pow(b); },
      [](T a, T b) { return (a + b).exp(); },
      [](T a, T b) { return (4 + a + b).log(); },
      [](T a, T b) { return (a + b).atan(); },
      [](T a, T b) { return (a.exp() + b.exp()).log(); },
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
  // Triune, or Num3<Real>, directly computes the primal and first and second
  // derivatives. We want to test it by comparing its behavior to the much
  // simpler and well-tested Num2, which directly computes the first derivative.
  // Num2 can be used to compute the second derivative by nesting.
  // Num2<Num2<Real>>, which we also call DualDual, computes both derivatives by
  // composition.
  auto f1s = functions<Triune>();
  auto f2s = functions<DualDual>();

  // Initialize a random number generator with its default seed
  // (deterministically).
  std::mt19937 g;
  std::uniform_real_distribution<double> unif(-2.0, 2.0);

  for (int i = 0, n = f1s.size(); i < n; i++) {
    double k1 = unif(g);
    double k2 = unif(g);
    double k3 = unif(g);
    double k4 = unif(g);
    double k5 = unif(g);
    double k6 = unif(g);

    // Construct two Triune values with the given downstream gradients.
    Triune t1 = Triune{k1, k2, k3};
    Triune t2 = Triune{k4, k5, k6};

    // Compute two DualDual values with the given downstream gradients.
    DualDual d1 = DualDual{Dual{k1, k2}, Dual{k2, k3}};
    DualDual d2 = DualDual{Dual{k4, k5}, Dual{k5, k6}};

    // Compute the function both ways
    auto f1 = f1s[i];
    auto f2 = f2s[i];
    auto t3 = f1(t1, t2);
    auto d3 = f2(d1, d2);

    // Compare the computed primal
    EXPECT_CLOSE(t3.as_double(), d3.as_double());

    // Compare the computed first derivative.
    // Note that DualDual stores the first derivative twice, so we test both
    // values.
    EXPECT_CLOSE(t3.derivative1.as_double(), d3.derivative1.as_double());
    EXPECT_CLOSE(t3.derivative1.as_double(), d3.primal.derivative1.as_double());

    // Compare the computed second derivative.
    EXPECT_CLOSE(
        t3.derivative2.as_double(), d3.derivative1.derivative1.as_double());
  }
}
