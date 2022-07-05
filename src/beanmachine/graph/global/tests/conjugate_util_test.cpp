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
    mean += samples[i][index]._double;
  }
  mean /= samples.size();
  return mean;
}

void add_gamma_gamma_conjugate_(
    Graph& g,
    double alpha_0,
    double beta_0,
    double alpha,
    Eigen::VectorXd x_observed) {
  /*
  beta ~ Gamma(alpha_0, beta_0)
  x ~ Gamma(alpha, beta)
  x is observed
  */
  uint alpha_0_node = g.add_constant_pos_real(alpha_0);
  uint beta_0_node = g.add_constant_pos_real(beta_0);
  uint alpha_node = g.add_constant_pos_real(alpha);

  uint gamma_dist = g.add_distribution(
      DistributionType::GAMMA,
      AtomicType::POS_REAL,
      {alpha_0_node, beta_0_node});
  uint beta_node = g.add_operator(OperatorType::SAMPLE, {gamma_dist});

  uint gamma_gamma_dist = g.add_distribution(
      DistributionType::GAMMA, AtomicType::POS_REAL, {alpha_node, beta_node});

  for (uint i = 0; i < x_observed.size(); i++) {
    uint x_node = g.add_operator(OperatorType::SAMPLE, {gamma_gamma_dist});
    g.observe(x_node, x_observed[i]);
  }

  g.query(beta_node);
}

double compute_gamma_gamma_moments_(
    double alpha_0,
    double beta_0,
    double alpha,
    Eigen::VectorXd x_observed) {
  double posterior_alpha = alpha_0 + alpha;
  double posterior_beta = beta_0 + x_observed.sum();
  double expected_mean = posterior_alpha / posterior_beta;
  return expected_mean;
}

std::vector<double> build_gamma_gamma_model(Graph& g) {
  /*
  Dimension 1:
  alpha_0 = 2.0
  beta_0 = 2.0
  alpha = 1.5

  beta ~ Gamma(alpha_0, beta_0)
  x ~ Gamma(alpha, beta)

  x observed as 2.0

  exact conjugate posterior is
  Gamma(alpha_0 + alpha, beta_0 + x) = Gamma(3.5, 4)

  expected mean is 3.5 / 4 = 0.875

  Dimension 2:
  alpha_0 = 0.5
  beta_0 = 1.0
  alpha = 0.5

  beta ~ Gamma(alpha_0, beta_0)
  x ~ Gamma(alpha, beta)

  x observed as 0.25

  exact conjugate posterior is
  Gamma(alpha_0 + alpha, beta_0 + x) = Gamma(1, 1.25)

  expected mean is 1.0 / 1.25 = 0.8
  https://en.wikipedia.org/wiki/Conjugate_prior
  */
  std::vector<double> expected_moments = {};

  // Dimension one
  double alpha_0 = 2.0;
  double beta_0 = 2.0;
  double alpha = 1.5;
  Eigen::VectorXd xs(1);
  xs << 2.0;
  add_gamma_gamma_conjugate_(g, alpha_0, beta_0, alpha, xs);
  expected_moments.push_back(
      compute_gamma_gamma_moments_(alpha_0, beta_0, alpha, xs));

  // Dimension 2
  alpha_0 = 0.5;
  beta_0 = 1.0;
  alpha = 0.5;
  xs << 0.25;
  add_gamma_gamma_conjugate_(g, alpha_0, beta_0, alpha, xs);
  expected_moments.push_back(
      compute_gamma_gamma_moments_(alpha_0, beta_0, alpha, xs));

  return expected_moments;
}

void add_normal_normal_conjugate_(
    Graph& g,
    double mu_0,
    double sigma_0,
    double sigma,
    Eigen::VectorXd x_observed) {
  /*
  mu ~ Normal(mu_0, sigma_0)
  x ~ Normal(mu, sigma)
  x is observed
  */
  uint mu_0_node = g.add_constant(mu_0);
  uint sigma_0_node = g.add_constant_pos_real(sigma_0);
  uint sigma_node = g.add_constant_pos_real(sigma);

  uint norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {mu_0_node, sigma_0_node});
  uint sample = g.add_operator(OperatorType::SAMPLE, {norm_dist});

  uint norm_norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {sample, sigma_node});

  for (uint i = 0; i < x_observed.size(); i++) {
    uint x_node = g.add_operator(OperatorType::SAMPLE, {norm_norm_dist});
    g.observe(x_node, x_observed[i]);
  }

  g.query(sample);
}

double compute_normal_normal_moments_(
    double mu_0,
    double sigma_0,
    double sigma,
    Eigen::VectorXd x_observed) {
  double sigma_0_sq = sigma_0 * sigma_0;
  double sigma_sq = sigma * sigma;
  double posterior_mu = 1 / (1 / sigma_0_sq + x_observed.size() / sigma_sq) *
      (mu_0 / sigma_0_sq + x_observed.sum() / sigma_sq);
  double expected_mean = posterior_mu;
  return expected_mean;
}

std::vector<double> build_normal_normal_model(Graph& g) {
  /*
  Dimension 1:
  mu_0 = 0
  sigma_0 = 2.0
  sigma = 1.0

  mu ~ Normal(mu_0, sigma_0)
  x_1, x_2 ~ Normal(mu, sigma)

  x_1 observed as 0.5
  x_2 observed as 1.5

  exact conjugate posterior is
  Normal(8/9, 4/9)

  expected mean is 8/9

  Dimension 2:
  mu_0 = 3.0
  sigma_0 = 1.0
  sigma = 10.0

  mu ~ Normal(mu_0, sigma_0)
  x_1, x_2 ~ Normal(mu, sigma)

  x_1 observed as 4.0
  x_2 observed as 6.0

  exact conjugate posterior is
  1 / (1 / 1 + 2 / 100) * (3 + 10 / 100)
  3.1 / (1.02)
  Normal(3.039, 0.98)

  expected mean is 3.039
  https://en.wikipedia.org/wiki/Conjugate_prior
  */
  std::vector<double> expected_moments = {};

  // Dim 1
  double mu_0 = 0.0;
  double sigma_0 = 2.0;
  double sigma = 1.0;
  Eigen::VectorXd xs(2);
  xs << 0.5, 1.5;
  add_normal_normal_conjugate_(g, mu_0, sigma_0, sigma, xs);
  expected_moments.push_back(
      compute_normal_normal_moments_(mu_0, sigma_0, sigma, xs));

  // Dim 2
  mu_0 = 3.0;
  sigma_0 = 1.0;
  sigma = 10.0;
  xs << 4.0, 6.0;
  add_normal_normal_conjugate_(g, mu_0, sigma_0, sigma, xs);
  expected_moments.push_back(
      compute_normal_normal_moments_(mu_0, sigma_0, sigma, xs));

  return expected_moments;
}

void add_gamma_normal_conjugate_(
    Graph& g,
    double alpha,
    double beta,
    double mu,
    Eigen::VectorXd x_observed) {
  /*
  gamma_sample ~ Gamma(alpha, beta)
  x_1, x_2 ~ Normal(mu, 1 / sqrt(gamma_sample))
  */
  uint alpha_node = g.add_constant_pos_real(alpha);
  uint beta_node = g.add_constant_pos_real(beta);

  uint gamma_dist = g.add_distribution(
      DistributionType::GAMMA, AtomicType::POS_REAL, {alpha_node, beta_node});
  uint sample = g.add_operator(OperatorType::SAMPLE, {gamma_dist});

  // 1 / sqrt(sample)
  uint point_five = g.add_constant(0.5);
  uint sqrt_sample = g.add_operator(OperatorType::POW, {sample, point_five});
  uint neg_one = g.add_constant(-1.0);
  uint inv_sqrt_sample =
      g.add_operator(OperatorType::POW, {sqrt_sample, neg_one});
  inv_sqrt_sample =
      g.add_operator(OperatorType::TO_POS_REAL, {inv_sqrt_sample});

  uint mu_node = g.add_constant(mu);
  uint gamma_norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {mu_node, inv_sqrt_sample});
  for (uint i = 0; i < x_observed.size(); i++) {
    uint x_node = g.add_operator(OperatorType::SAMPLE, {gamma_norm_dist});
    g.observe(x_node, x_observed[i]);
  }

  g.query(sample);
}

double compute_gamma_normal_moments_(
    double alpha,
    double beta,
    double mu,
    Eigen::VectorXd x_observed) {
  double posterior_alpha = alpha + x_observed.size() / 2;
  double posterior_beta =
      beta + Eigen::pow((x_observed.array() - mu), 2).sum() / 2;
  double expected_mean = posterior_alpha / posterior_beta;
  return expected_mean;
}

std::vector<double> build_gamma_normal_model(Graph& g) {
  /*
  alpha = 2.0
  beta = 1.0
  mu = 5.0

  gamma_sample ~ Gamma(alpha, beta)
  x_1, x_2 ~ Normal(mu, 1 / sqrt(gamma_sample))

  x_1 observed as 6.0
  x_2 observed as 7.0

  exact conjugate posterior is
  Gamma(3.0, 3.5)

  expected mean is 3.0 / 3.5
  https://en.wikipedia.org/wiki/Conjugate_prior
  */
  std::vector<double> expected_moments;

  double alpha = 2.0;
  double beta = 1.0;
  double mu = 5.0;
  Eigen::VectorXd xs(2);
  xs << 6.0, 7.0;
  add_gamma_normal_conjugate_(g, alpha, beta, mu, xs);
  expected_moments.push_back(
      compute_gamma_normal_moments_(alpha, beta, mu, xs));

  return expected_moments;
}

void add_beta_binomial_model_(
    Graph& g,
    double alpha,
    double beta,
    int n,
    Eigen::VectorXi xs) {
  uint alpha_node = g.add_constant_pos_real(alpha);
  uint beta_node = g.add_constant_pos_real(beta);

  uint beta_dist = g.add_distribution(
      DistributionType::BETA, AtomicType::PROBABILITY, {alpha_node, beta_node});
  uint p = g.add_operator(OperatorType::SAMPLE, {beta_dist});

  uint n_node = g.add_constant((natural_t)n);
  uint beta_binomial_dist = g.add_distribution(
      DistributionType::BINOMIAL, AtomicType::NATURAL, {n_node, p});
  uint obs = g.add_operator(OperatorType::SAMPLE, {beta_binomial_dist});

  for (uint i = 0; i < xs.size(); i++) {
    g.observe(obs, (natural_t)xs[i]);
  }
  g.query(p);
}

double compute_beta_binomial_moments_(
    double alpha,
    double beta,
    int n,
    Eigen::VectorXi xs) {
  double posterior_alpha = alpha + xs.sum();
  double posterior_beta = beta + n - xs.sum();
  double expected_mean = posterior_alpha / (posterior_alpha + posterior_beta);
  return expected_mean;
}

std::vector<double> build_beta_binomial_model(Graph& g) {
  /*
  alpha = 3.0
  beta = 2.0
  n = 3

  p ~ Beta(alpha, beta)
  k ~ Binomial(n, p)

  k observed as 2

  exact conjugate posterior is
  Beta(5.0, 3.0)

  expected mean is 0.625
  https://en.wikipedia.org/wiki/Conjugate_prior
  */
  std::vector<double> expected_moments;

  // should be same values as test_beta_binomial_model
  double alpha = 3.0;
  double beta = 2.0;
  int n = 3;
  Eigen::VectorXi xs(1);
  xs << 2;
  add_beta_binomial_model_(g, alpha, beta, n, xs);
  expected_moments.push_back(
      compute_beta_binomial_moments_(alpha, beta, n, xs));

  return expected_moments;
}

void test_conjugate_model_moments(
    GlobalMH& mh,
    std::vector<double> expected_moments,
    int num_samples,
    int num_warmup_samples,
    double delta,
    int seed) {
  std::vector<std::vector<NodeValue>> samples =
      mh.infer(num_samples, seed, num_warmup_samples);
  EXPECT_EQ(samples.size(), num_samples);
  for (uint i = 0; i < expected_moments.size(); i++) {
    EXPECT_NEAR(_compute_mean_at_index(samples, i), expected_moments[i], delta);
  }
}

} // namespace graph
} // namespace beanmachine
