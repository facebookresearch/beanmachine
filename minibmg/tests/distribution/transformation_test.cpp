/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <beanmachine/minibmg/tests/test_utils.h>
#include "beanmachine/minibmg/ad/num2.h"
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/distribution/distribution.h"
#include "beanmachine/minibmg/distribution/make_distribution.h"
#include "beanmachine/minibmg/distribution/transformation.h"
#include "beanmachine/minibmg/operator.h"

// using namespace ::testing;
using namespace ::beanmachine::minibmg;
using namespace ::std;
using Dual = Num2<Real>;

namespace {

// The logit function
template <class T>
requires Number<T> T log_transform(T x) {
  return log(x);
}

// The first derivative of the logit function
template <class T>
requires Number<T> T log_transform1(T x) {
  return 1 / x;
}

} // namespace

TEST(transforms, half_normal) {
  // Test the default transform (log) on a half normal distribution.
  Dual stddev = {3, 1.1};
  auto get_parameter = [=](unsigned) { return stddev; };
  DistributionPtr<Dual> d = make_distribution<Dual>(
      Operator::DISTRIBUTION_HALF_NORMAL, get_parameter);
  double sample = 4.5;
  // Samples and observations do not have gradients, so we use none here.
  double sample_grad = 0;
  Dual constrained_sample{sample, sample_grad}; // a hypothetical sample
  Dual untransformed_log_prob = d->log_prob(constrained_sample);
  TransformationPtr<Dual> transform = d->transformation();
  Dual unconstrained_sample = transform->call(constrained_sample);
  Dual expected_unconstrained = log_transform(constrained_sample);
  EXPECT_CLOSE(
      expected_unconstrained.as_double(), unconstrained_sample.as_double());
  EXPECT_CLOSE(
      transform->inverse(unconstrained_sample).as_double(),
      constrained_sample.as_double());
  EXPECT_CLOSE(
      transform->inverse(unconstrained_sample).derivative1.as_double(),
      constrained_sample.derivative1.as_double());
  EXPECT_CLOSE(
      expected_unconstrained.derivative1.as_double(),
      unconstrained_sample.derivative1.as_double());
  auto transformed_log_prob =
      transform->transform_log_prob(constrained_sample, untransformed_log_prob);
  auto expected_transformed_log_prob =
      untransformed_log_prob + log(log_transform1(constrained_sample));
  EXPECT_CLOSE(
      expected_transformed_log_prob.as_double(),
      transformed_log_prob.as_double());
  EXPECT_CLOSE(
      expected_transformed_log_prob.derivative1.as_double(),
      transformed_log_prob.derivative1.as_double());
}

namespace {

// The logit function
template <class T>
requires Number<T> T logit(T p) {
  return log(p / (1 - p));
}

// The first derivative of the logit function
template <class T>
requires Number<T> T logit1(T p) {
  return 1 / (p - p * p);
}

} // namespace

TEST(transforms, beta) {
  // TODO: Test the default transform (logit) on a beta distribution
  Dual a{2, 0.12};
  Dual b{3, 0.21};

  auto get_parameter = [&](unsigned i) { return (i == 0) ? a : b; };
  DistributionPtr<Dual> d =
      make_distribution<Dual>(Operator::DISTRIBUTION_BETA, get_parameter);
  // Samples and observations do not have gradients, so we use none here.
  double sample = 0.14;
  Dual constrained_sample{sample, 0}; // a hypothetical sample; zero gradient
  Dual untransformed_log_prob = d->log_prob(constrained_sample);
  TransformationPtr<Dual> transform = d->transformation();
  Dual unconstrained_sample = transform->call(constrained_sample);
  Dual expected_unconstrained = logit(constrained_sample);
  EXPECT_CLOSE(
      expected_unconstrained.as_double(), unconstrained_sample.as_double());
  EXPECT_CLOSE(
      transform->inverse(unconstrained_sample).as_double(),
      constrained_sample.as_double());
  EXPECT_CLOSE(
      transform->inverse(unconstrained_sample).derivative1.as_double(),
      constrained_sample.derivative1.as_double());
  EXPECT_CLOSE(
      expected_unconstrained.derivative1.as_double(),
      unconstrained_sample.derivative1.as_double());
  auto transformed_log_prob =
      transform->transform_log_prob(constrained_sample, untransformed_log_prob);
  auto expected_transformed_log_prob =
      untransformed_log_prob + log(logit1(constrained_sample));
  EXPECT_CLOSE(
      expected_transformed_log_prob.as_double(),
      transformed_log_prob.as_double());
  EXPECT_CLOSE(
      expected_transformed_log_prob.derivative1.as_double(),
      transformed_log_prob.derivative1.as_double());
}

TEST(transforms, normal) {
  auto get_parameter = [=](unsigned) { return 1.0; };
  DistributionPtr<Dual> d =
      make_distribution<Dual>(Operator::DISTRIBUTION_NORMAL, get_parameter);
  TransformationPtr<Dual> transform = d->transformation();
  EXPECT_EQ(transform, nullptr);
}
