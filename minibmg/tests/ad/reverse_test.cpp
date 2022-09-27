/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <random>
#include "beanmachine/minibmg/ad/num2.h"
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/ad/reverse.h"
#include "beanmachine/minibmg/tests/test_utils.h"

using namespace ::testing;
using namespace beanmachine::minibmg;

using Back = Reverse<Real>;
using Fwd = Num2<Real>;

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

// Compare the behavior of Reverse<Real> (aka Back) to Num2<Real> (aka Fwd).
// Assuming that Fwd is well tested and correct, this offers evidence of the
// correctness of Back.  Both should get the same result when computing the same
// scalar gradient.  That might not be the case for matrices, because they may
// differ by a transpose.
TEST(reverse_test, compare_to_num2) {
  // Compare functions computed in reverse mode...
  auto f1s = functions<Back>();

  // To the same functions computed in forward mode.
  auto f2s = functions<Fwd>();

  // Use a random number generator with its default seed.
  std::mt19937 g;

  // Generate several doubles between -2.0 and 2.0.
  std::uniform_real_distribution<double> unif(-2.0, 2.0);

  // Test each function 5 times with different sample values
  for (int k = 0; k < 5; k++) {
    // For arguments a and b, we take random values for...
    double a0 = unif(g); // a's value
    double a1 = unif(g); // a's gradient
    double b0 = unif(g); // b's value
    double b1 = unif(g); // b's gradient

    for (int i = 0, n = f1s.size(); i < n; i++) {
      // df/da and df/db reverse (both at once!)
      const auto& f1 = f1s[i];
      Back a_back = a0;
      Back b_back = b0;
      Back result_back = f1(a_back, b_back);
      Real primal_back = result_back.ptr->primal;
      result_back.reverse(1);
      Real dfda_back = a1 * a_back.ptr->adjoint;
      Real dfdb_back = b1 * b_back.ptr->adjoint;

      // compute the derivative of the result of the function with respect to
      // its first input in forward mode
      {
        // df/da forward
        const auto& f2 = f2s[i];
        Fwd a_fwd = Fwd{a0, a1};
        Fwd b_fwd = b0;
        Fwd result_fwd = f2(a_fwd, b_fwd);
        Real primal_fwd = result_fwd.primal;
        Real dfda_fwd = result_fwd.derivative1;

        EXPECT_CLOSE(primal_fwd.as_double(), primal_back.as_double());
        EXPECT_CLOSE(dfda_fwd.as_double(), dfda_back.as_double());
      }

      // compute the derivative of the result of the function with respect to
      // its second input in forward mode
      {
        // df/db forward
        const auto& f2 = f2s[i];
        Fwd a_fwd = a0;
        Fwd b_fwd = Fwd{b0, b1};
        Fwd result_fwd = f2(a_fwd, b_fwd);
        Real primal_fwd = result_fwd.primal;
        Real dfdb_fwd = result_fwd.derivative1;

        EXPECT_CLOSE(primal_fwd.as_double(), primal_back.as_double());
        EXPECT_CLOSE(dfdb_fwd.as_double(), dfdb_back.as_double());
      }
    }
  }
}
