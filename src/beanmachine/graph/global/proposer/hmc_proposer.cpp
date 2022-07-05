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

  int size = static_cast<int>(position.size());
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
    std::mt19937& gen,
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
      find_reasonable_step_size(state, gen, position);
      step_size_adapter.initialize(step_size);
    }
  }

  if (iteration == num_warmup_samples) {
    step_size = step_size_adapter.finalize_step_size();
  }
}

// Follows Algorithm 4 of NUTS paper
void HmcProposer::find_reasonable_step_size(
    GlobalState& state,
    std::mt19937& gen,
    Eigen::VectorXd position) {
  Eigen::VectorXd initial_position = position;
  Eigen::VectorXd momentum = initialize_momentum(position, gen);
  const double LOG_CONSTANT = std::log(0.8);
  // 0.8 taken from Stan
  // https://github.com/stan-dev/stan/blob/86f018b106303001194c7760d7d69a9ebc5557da/src/stan/mcmc/hmc/base_hmc.hpp#L104
  double acceptance_log_prob =
      compute_new_step_acceptance_probability(state, position, momentum);
  int direction = 1;
  if (std::isnan(acceptance_log_prob) or acceptance_log_prob < LOG_CONSTANT) {
    direction = -1;
  }
  int max_step_adjustments = 20;
  // TODO: add errors/warnings when step size becomes too large or too small
  for (int i = 0; i < max_step_adjustments; i++) {
    momentum = initialize_momentum(position, gen);
    double prev_step_size = step_size;
    step_size = std::pow(2, direction) * step_size;
    acceptance_log_prob =
        compute_new_step_acceptance_probability(state, position, momentum);

    // don't increase step_size if acceptance is NaN
    if (std::isnan(acceptance_log_prob) and direction == 1) {
      step_size = prev_step_size;
      break;
    }

    // break when acceptance prob crosses threshold
    if ((direction == 1 and acceptance_log_prob <= LOG_CONSTANT) or
        (direction == -1 and acceptance_log_prob >= LOG_CONSTANT)) {
      break;
    }
  }
  state.set_flattened_unconstrained_values(initial_position);
}

double HmcProposer::compute_new_step_acceptance_probability(
    GlobalState& state,
    Eigen::VectorXd position,
    Eigen::VectorXd momentum) {
  double current_H = compute_hamiltonian(state, position, momentum);

  double direction = 1.0;
  std::vector<Eigen::VectorXd> leapfrog_result =
      leapfrog(state, position, momentum, direction);
  Eigen::VectorXd position_new = leapfrog_result[0];
  Eigen::VectorXd momentum_new = leapfrog_result[1];

  double proposed_H = compute_hamiltonian(state, position_new, momentum_new);

  return current_H - proposed_H;
}

double HmcProposer::compute_hamiltonian(
    GlobalState& state,
    Eigen::VectorXd position,
    Eigen::VectorXd momentum) {
  double K = compute_kinetic_energy(momentum);
  state.set_flattened_unconstrained_values(position);
  state.update_log_prob();
  double U = -state.get_log_prob();
  return K + U;
}

std::vector<Eigen::VectorXd> HmcProposer::leapfrog(
    GlobalState& state,
    Eigen::VectorXd position,
    Eigen::VectorXd momentum,
    double direction) {
  // momentum half-step
  state.set_flattened_unconstrained_values(position);
  Eigen::VectorXd grad_U = compute_potential_gradient(state);
  momentum = momentum - direction * step_size * grad_U / 2;
  // position full-step
  Eigen::VectorXd mass_momentum =
      mass_inv.diagonal().array() * momentum.array();
  position = position + direction * step_size * mass_momentum;
  // momentum half-step
  state.set_flattened_unconstrained_values(position);
  grad_U = compute_potential_gradient(state);
  momentum = momentum - direction * step_size * grad_U / 2;

  return {position, momentum};
}

Eigen::VectorXd HmcProposer::initialize_momentum(
    Eigen::VectorXd position,
    std::mt19937& gen) {
  Eigen::VectorXd momentum(static_cast<int>(position.size()));
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

  int num_steps = static_cast<int>(std::min(
      static_cast<double>(max_steps), (ceil(path_length / step_size))));

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
