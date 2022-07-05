/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/proposer/hmc_util.h"

namespace beanmachine {
namespace graph {

StepSizeAdapter::StepSizeAdapter(double optimal_acceptance_prob) {
  this->optimal_acceptance_prob = optimal_acceptance_prob;
}

void StepSizeAdapter::initialize(double step_size) {
  gamma = 0.05;
  t = 10.0;
  kappa = 0.75;
  mu = std::log(10.0 * step_size);
  log_best_step_size = 0;
  closeness = 0.0;
}

double StepSizeAdapter::update_step_size(
    int iteration,
    double acceptance_prob) {
  double iter = (double)iteration;

  double closeness_frac = 1.0 / (iter + t);
  closeness = (1 - closeness_frac) * closeness +
      (closeness_frac * (optimal_acceptance_prob - acceptance_prob));

  double log_step_size = mu - std::sqrt(iter) / gamma * closeness;

  double step_frac = std::pow(iter, -kappa);
  log_best_step_size =
      (step_frac * log_step_size) + (1 - step_frac) * log_best_step_size;

  return std::exp(log_step_size);
}

double StepSizeAdapter::finalize_step_size() {
  return std::exp(log_best_step_size);
}

void DiagonalCovarianceComputer::initialize(int size) {
  iteration = 0;
  M2 = Eigen::MatrixXd::Zero(size, size);
  sample_mean = Eigen::VectorXd::Zero(size);
}

void DiagonalCovarianceComputer::reset() {
  initialize(static_cast<int>(M2.rows()));
}

void DiagonalCovarianceComputer::update(Eigen::VectorXd sample) {
  // uses Welford's online algorithm
  iteration++;
  Eigen::VectorXd delta = sample - sample_mean;
  sample_mean += delta / iteration;
  Eigen::VectorXd delta2 = sample - sample_mean;
  M2.diagonal() += delta.cwiseProduct(delta2);
}

Eigen::MatrixXd DiagonalCovarianceComputer::finalize_updates() {
  Eigen::MatrixXd covariance = M2 / (iteration - 1);
  // regularization as seen in Stan+
  // https://github.com/stan-dev/stan/blob/b7faab65a9db2b8767047bcf7320214b185295d7/src/stan/mcmc/covar_adaptation.hpp
  double weight = iteration / (iteration + 5.0);
  double regularization_constant = 1e-3;
  covariance = weight * covariance +
      regularization_constant * (1 - weight) *
          Eigen::MatrixXd::Identity(covariance.rows(), covariance.cols());
  return covariance;
}

void WindowedMassMatrixAdapter::initialize(int num_warmup_samples, int size) {
  // warmup as seen in Stan
  // https://github.com/stan-dev/stan/blob/b7faab65a9db2b8767047bcf7320214b185295d7/src/stan/mcmc/windowed_adaptation.hpp
  const int minimum_samples_for_adaptation = 20;
  const int minimum_samples_for_full_adaptation = 150;
  if (num_warmup_samples < minimum_samples_for_adaptation) {
    // no adaptation intervals
    start_window_iter = num_warmup_samples + 1;
  } else if (num_warmup_samples < minimum_samples_for_full_adaptation) {
    // not enough samples for regular interval sizes
    start_window_iter = (int)(0.15 * num_warmup_samples);
    end_adaptation_iter = (int)(0.9 * num_warmup_samples);
    window_size = end_adaptation_iter - start_window_iter;
  } else {
    start_window_iter = 75;
    end_adaptation_iter = num_warmup_samples - 50;
    window_size = 25;
  }

  mass_inv = Eigen::MatrixXd::Identity(size, size);

  cov_alg.initialize(size);
}

bool WindowedMassMatrixAdapter::is_end_window(int iteration) {
  return (iteration == start_window_iter + window_size);
}

void WindowedMassMatrixAdapter::get_mass_matrix_and_reset(
    int iteration,
    Eigen::MatrixXd& mass_inv) {
  mass_inv = cov_alg.finalize_updates();
  cov_alg.reset();
  start_window_iter = iteration;
  window_size = 2 * window_size;
  if (end_adaptation_iter - iteration < window_size * 2) {
    window_size = end_adaptation_iter - iteration;
  }
}

void WindowedMassMatrixAdapter::update_mass_matrix(
    int iteration,
    Eigen::VectorXd sample) {
  if (iteration <= start_window_iter) {
    return;
  }
  cov_alg.update(sample);
}

} // namespace graph
} // namespace beanmachine
