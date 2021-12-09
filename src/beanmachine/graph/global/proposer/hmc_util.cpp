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
  iteration = 0;
}

double StepSizeAdapter::update_step_size(double acceptance_prob) {
  iteration++;
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

} // namespace graph
} // namespace beanmachine
