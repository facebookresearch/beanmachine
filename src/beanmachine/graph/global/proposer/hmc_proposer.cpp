/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/proposer/hmc_proposer.h"

namespace beanmachine {
namespace graph {

HmcProposer::HmcProposer(
    double path_length,
    double step_size,
    double optimal_acceptance_prob)
    : GlobalProposer(),
      step_size_adapter(StepSizeAdapter(optimal_acceptance_prob)) {
  this->path_length = path_length;
  this->step_size = step_size;
}

void HmcProposer::initialize(
    GlobalState& /*state*/,
    std::mt19937& /*gen*/,
    int num_warmup_samples) {
  if (num_warmup_samples > 0) {
    step_size_adapter.initialize(step_size);
  }
}

double HmcProposer::compute_kinetic_energy(Eigen::VectorXd momentum) {
  return 0.5 * momentum.dot(momentum);
}

Eigen::VectorXd HmcProposer::compute_potential_gradient(GlobalState& state) {
  state.update_backgrad();
  Eigen::VectorXd grad1;
  state.get_flattened_unconstrained_grads(grad1);
  return -grad1;
}

void HmcProposer::warmup(
    double acceptance_prob,
    int iteration,
    int num_warmup_samples) {
  if (iteration < num_warmup_samples) {
    step_size = step_size_adapter.update_step_size(acceptance_prob);
  } else {
    step_size = step_size_adapter.finalize_step_size();
  }
}

Eigen::VectorXd HmcProposer::initialize_momentum(
    Eigen::VectorXd position,
    std::mt19937& gen) {
  Eigen::VectorXd momentum(position.size());
  std::normal_distribution<double> normal_dist(0.0, 1.0);
  for (int i = 0; i < momentum.size(); i++) {
    momentum[i] = normal_dist(gen);
  }

  return momentum;
}

double HmcProposer::propose(GlobalState& state, std::mt19937& gen) {
  Eigen::VectorXd position;
  state.get_flattened_unconstrained_values(position);
  state.update_log_prob();
  double initial_U = -state.get_log_prob();

  Eigen::VectorXd momentum(position.size());
  std::normal_distribution<double> dist(0.0, 1.0);
  for (int i = 0; i < momentum.size(); i++) {
    momentum[i] = dist(gen);
  }
  double initial_K = compute_kinetic_energy(momentum);

  int num_steps = static_cast<int>(ceil(path_length / step_size));

  // momentum half-step
  Eigen::VectorXd grad_U = compute_potential_gradient(state);
  momentum = momentum - step_size * grad_U / 2;
  for (int i = 0; i < num_steps; i++) {
    // position full-step
    position = position + step_size * momentum;

    // momentum step
    state.set_flattened_unconstrained_values(position);
    grad_U = compute_potential_gradient(state);
    if (i < num_steps - 1) {
      // full-step
      momentum = momentum - step_size * grad_U;
    } else {
      // half-step at the last iteration
      momentum = momentum - step_size * grad_U / 2;
    }
  }

  double final_K = compute_kinetic_energy(momentum);
  state.update_log_prob();
  double final_U = -state.get_log_prob();
  return initial_U - final_U + initial_K - final_K;
}

} // namespace graph
} // namespace beanmachine
