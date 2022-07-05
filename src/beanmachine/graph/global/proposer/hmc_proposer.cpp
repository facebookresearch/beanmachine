/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/proposer/hmc_proposer.h"
#include <algorithm>

namespace beanmachine {
namespace graph {

HmcProposer::HmcProposer(
    double path_length,
    double step_size,
    bool adapt_mass_matrix,
    double optimal_acceptance_prob)
    : GlobalProposer(),
      step_size_adapter(StepSizeAdapter(optimal_acceptance_prob)),
      mass_matrix_adapter(WindowedMassMatrixAdapter()) {
  this->path_length = path_length;
  this->step_size = step_size;
  this->adapt_mass_matrix = adapt_mass_matrix;
  max_steps = 1000;
}

void HmcProposer::initialize(
    GlobalState& state,
    std::mt19937& /*gen*/,
    int num_warmup_samples) {
  Eigen::VectorXd position;
  state.get_flattened_unconstrained_values(position);

  int size = position.size();
  mass_inv = Eigen::MatrixXd::Identity(size, size);
  mass_matrix_diagonal = Eigen::ArrayXd::Ones(size);

  if (num_warmup_samples > 0) {
    step_size_adapter.initialize(step_size);
    if (adapt_mass_matrix) {
      mass_matrix_adapter.initialize(num_warmup_samples, size);
    }
  }
}

double HmcProposer::compute_kinetic_energy(Eigen::VectorXd momentum) {
  Eigen::VectorXd mass_momentum =
      momentum.array() * mass_inv.diagonal().array();
  return 0.5 * (mass_momentum.dot(momentum));
}

Eigen::VectorXd HmcProposer::compute_potential_gradient(GlobalState& state) {
  state.update_backgrad();
  Eigen::VectorXd grad1;
  state.get_flattened_unconstrained_grads(grad1);
  return -grad1;
}

void HmcProposer::warmup(
    GlobalState& state,
    std::mt19937& /*gen*/,
    double acceptance_prob,
    int iteration,
    int num_warmup_samples) {
  step_size = step_size_adapter.update_step_size(iteration, acceptance_prob);

  if (adapt_mass_matrix) {
    Eigen::VectorXd sample;
    state.get_flattened_unconstrained_values(sample);
    mass_matrix_adapter.update_mass_matrix(iteration, sample);
    bool window_end = mass_matrix_adapter.is_end_window(iteration);

    if (window_end) {
      mass_matrix_adapter.get_mass_matrix_and_reset(iteration, mass_inv);
      mass_matrix_diagonal = mass_inv.diagonal().array().sqrt().inverse();
      Eigen::VectorXd position;
      state.get_flattened_unconstrained_values(position);
      // TODO: update in next diff
      // find_reasonable_step_size(state, gen, position);
      step_size_adapter.initialize(step_size);
    }
  }

  if (iteration == num_warmup_samples) {
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

  return momentum.array() * mass_matrix_diagonal;
}

double HmcProposer::propose(GlobalState& state, std::mt19937& gen) {
  Eigen::VectorXd position;
  state.get_flattened_unconstrained_values(position);
  state.update_log_prob();
  double initial_U = -state.get_log_prob();

  Eigen::VectorXd momentum = initialize_momentum(position, gen);
  double initial_K = compute_kinetic_energy(momentum);

  int num_steps =
      std::min(max_steps, static_cast<int>(ceil(path_length / step_size)));

  // momentum half-step
  Eigen::VectorXd grad_U = compute_potential_gradient(state);
  momentum = momentum - step_size * grad_U / 2;
  for (int i = 0; i < num_steps; i++) {
    // position full-step
    Eigen::VectorXd mass_momentum =
        mass_inv.diagonal().array() * momentum.array();
    position = position + step_size * mass_momentum;

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
