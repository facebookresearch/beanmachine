/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <beanmachine/minibmg/tests/test_utils.h>
#include "beanmachine/minibmg/ad/num2.h"
#include "beanmachine/minibmg/ad/num3.h"
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/distribution/bernoulli.h"
#include "beanmachine/minibmg/distribution/beta.h"
#include "beanmachine/minibmg/distribution/normal.h"
#include "beanmachine/minibmg/log_prob.h"

// using namespace ::testing;
using namespace ::beanmachine::minibmg;
using namespace ::std;

TEST(log_prob, normal_real) {
  Real v{1.234};
  Real mean{4.223};
  Real stdev{6.221};
  // expected values computed using wolfram alpha:
  // log(PDF[NormalDistribution[4.223, 6.221], 1.234])
  double expected = -2.86229;
  EXPECT_CLOSE(expected, (Normal<Real>{mean, stdev}.log_prob(v).as_double()));
}

using Dual = Num2<Real>;
TEST(log_prob, normal_dual) {
  Dual v{1.234, 1};
  Dual mean{4.223};
  Dual stdev{6.221};
  // log(PDF[NormalDistribution[4.223, 6.221], 1.234])
  double expected = -2.86229;
  auto result = Normal<Dual>{mean, stdev}.log_prob(v);
  EXPECT_CLOSE(expected, result.as_double());
  // ReplaceAll[D[log(PDF[NormalDistribution[4.223, 6.221], x]), {x, 1}], {x
  // -> 1.234}]
  double expected_derivative1 = 0.0772335;
  EXPECT_CLOSE(expected_derivative1, result.derivative1.as_double());
}

using Triune = Num3<Real>;
TEST(log_prob, normal_triune) {
  Triune v{1.234, 1, 0};
  Triune mean{4.223};
  Triune stdev{6.221};
  // log(PDF[NormalDistribution[4.223, 6.221], 1.234])
  double expected = -2.86229;
  auto result = Normal<Triune>{mean, stdev}.log_prob(v);
  EXPECT_CLOSE(expected, result.as_double());
  double expected_derivative1 = 0.0772335;
  EXPECT_CLOSE(expected_derivative1, result.derivative1.as_double());
  // ReplaceAll[D[log(PDF[NormalDistribution[4.223, 6.221], x]), {x, 2}], {x
  // -> 1.234}]
  double expected_derivative2 = -0.0258392;
  EXPECT_CLOSE(expected_derivative2, result.derivative2.as_double());
}
