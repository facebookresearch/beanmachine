/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/tests/conjugate_util_test.h"
#include <gtest/gtest.h>
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

double _compute_mean_at_index(
    std::vector<std::vector<NodeValue>> samples,
    int index) {
  double mean = 0;
  for (int i = 0; i < samples.size(); i++) {
    mean += samples[i][index]._value;
  }
  mean /= samples.size();
  return mean;
}

void build_gamma_gamma_model(Graph& g) {
  /*
  alpha_0 = 2.0
  beta_0 = 2.0
  alpha = 1.5

  beta ~ Gamma(alpha_0, beta_0)
  x ~ Gamma(alpha, beta)

  x observed as 2.0

  exact conjugate posterior is
  Gamma(alpha_0 + alpha, beta_0 + 1 / x) = Gamma(3.5, 4)

  expected mean is 3.5 / 4 = 0.875
  https://en.wikipedia.org/wiki/Conjugate_prior
  */
  uint alpha_0 = g.add_constant_pos_real(2.0);
  uint beta_0 = g.add_constant_pos_real(2.0);
  uint alpha = g.add_constant_pos_real(1.5);

  uint gamma_dist = g.add_distribution(
      DistributionType::GAMMA, AtomicType::POS_REAL, {alpha_0, beta_0});
  uint beta = g.add_operator(OperatorType::SAMPLE, {gamma_dist});

  uint gamma_gamma_dist = g.add_distribution(
      DistributionType::GAMMA, AtomicType::POS_REAL, {alpha, beta});
  uint x = g.add_operator(OperatorType::SAMPLE, {gamma_gamma_dist});
  g.customize_transformation(TransformType::LOG, {beta});

  g.observe(x, 2.0);
  g.query(beta);
}

void test_gamma_gamma_model(
    GlobalMH& mh,
    int num_samples,
    int seed,
    int num_warmup_samples,
    double delta) {
  std::vector<std::vector<NodeValue>> samples =
      mh.infer(num_samples, seed, num_warmup_samples);
  EXPECT_EQ(samples.size(), num_samples);
  EXPECT_NEAR(_compute_mean_at_index(samples, 0), 0.875, delta);
}

void build_normal_normal_model(Graph& g) {
  /*
  mu_0 = 0
  sigma_0 = 2.0
  sigma = 1.0

  mu ~ Normal(mu_0, sigma_0)
  x_1, x_2 ~ Normal(mu, sigma)

  x_1 observed as 0.5
  x_2 observed as 1.5

  exact conjugate posterior is
  Normal(0.66.., 0.5)

  expected mean is 0.66..
  https://en.wikipedia.org/wiki/Conjugate_prior
  */

  uint zero = g.add_constant(0.0);
  uint one = g.add_constant_pos_real(1.0);

  uint norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {zero, one});
  uint sample = g.add_operator(OperatorType::SAMPLE, {norm_dist});

  uint norm_norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {sample, one});
  uint obs1 = g.add_operator(OperatorType::SAMPLE, {norm_norm_dist});
  uint obs2 = g.add_operator(OperatorType::SAMPLE, {norm_norm_dist});

  g.observe(obs1, 0.5);
  g.observe(obs2, 1.5);
  g.query(sample);
}

void test_normal_normal_model(
    GlobalMH& mh,
    int num_samples,
    int seed,
    int num_warmup_samples,
    double delta) {
  std::vector<std::vector<NodeValue>> samples =
      mh.infer(num_samples, seed, num_warmup_samples);
  EXPECT_EQ(samples.size(), num_samples);
  EXPECT_NEAR(_compute_mean_at_index(samples, 0), 2.0 / 3, delta);
}

} // namespace graph
} // namespace beanmachine
