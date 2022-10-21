/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <memory>
#include "beanmachine/minibmg/fluid_factory.h"
#include "beanmachine/minibmg/inference/hmc_world.h"
#include "beanmachine/minibmg/inference/mle_inference.h"

using namespace ::testing;
using namespace beanmachine::minibmg;

TEST(mle_inference_test, coin_flipping) {
  // Take a familiar-looking model and run gradient descent to confirm that we
  // get the expected value.
  Graph::FluidFactory f;
  auto d = beta(2, 2);
  auto s = sample(d);
  auto bn = bernoulli(s);
  f.observe(sample(bn), 1);
  f.observe(sample(bn), 1);
  f.observe(sample(bn), 0);
  f.query(s);

  auto inference_result = mle_inference_0(f.build());

  auto s_expected = 0.6;
  auto s_inferred = inference_result[0];

  ASSERT_NEAR(s_inferred, s_expected, 1e-7);
}

TEST(mle_inference_test, sqrt) {
  // Compute the square root of two using the the technique of "observing a
  // functional"
  // (https://fb.workplace.com/groups/pplxfn/permalink/3245784932349729/).
  Graph::FluidFactory f;
  auto n1 = normal(1, 1e7);
  auto sqrt = sample(n1);
  auto two = sqrt * sqrt;
  auto n2 = normal(two, 1);
  f.observe(sample(n2), 2);
  f.query(sqrt);

  const double learning_rate = 0.1;
  std::vector<double> initial_proposals{1.0};
  auto inference_result = mle_inference_0(
      f.build(),
      /* learning_rate= */ learning_rate,
      /* num_rounds= */ 25,
      /* initial_proposals= */ initial_proposals);

  auto s_expected = std::sqrt(2.0);
  auto s_inferred = inference_result[0];

  ASSERT_NEAR(s_inferred, s_expected, 1e-7);
}

TEST(mle_inference_test, normal) {
  // Test inference in the absence of observations.
  Graph::FluidFactory f;
  auto n = normal(2, 3);
  f.query(sample(n));
  // Learning rate set by trial and error
  const double learning_rate = 5;
  auto inference_result =
      mle_inference_0(f.build(), /* learning_rate= */ learning_rate);

  auto s_expected = 2.0;
  auto s_inferred = inference_result[0];

  ASSERT_NEAR(s_inferred, s_expected, 1e-7);
}

TEST(mle_inference_test, beta) {
  // Test inference in the absence of observations.
  Graph::FluidFactory f;
  double a = 7;
  double b = 5;
  auto n = beta(a, b);
  auto s = sample(n);
  f.query(s);

  // From https://en.wikipedia.org/wiki/Beta_distribution
  // The mode of a Beta distributed random variable X with α, β > 1 is the most
  // likely value of the distribution (corresponding to the peak in the PDF),
  // and is given by the following expression: (α - 1)/(α + β - 2).
  double mode = (a - 1) / (a + b - 2);
  auto s_expected = mode;

  auto inference_result = mle_inference_0(f.build());
  auto s_inferred = inference_result[0];

  ASSERT_NEAR(s_inferred, s_expected, 1e-7);
}
