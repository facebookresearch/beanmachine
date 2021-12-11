/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/proposer/nuts_proposer.h"

namespace beanmachine {
namespace graph {

NutsProposer::NutsProposer(double optimal_acceptance_prob)
    : HmcProposer(0.0, 1.0, optimal_acceptance_prob) {
  step_size = 1.0;
  delta_max = 1000;
  max_tree_depth = 10;
}

void NutsProposer::initialize(
    GlobalState& state,
    std::mt19937& gen,
    int /*num_warmup_samples*/) {
  Eigen::VectorXd position;
  state.get_flattened_unconstrained_values(position);
  find_reasonable_step_size(state, gen, position);
  step_size_adapter.initialize(step_size);
}

void NutsProposer::warmup(
    double /*acceptance_prob*/,
    int iteration,
    int num_warmup_samples) {
  step_size = step_size_adapter.update_step_size(warmup_acceptance_prob);
  if (iteration == num_warmup_samples) {
    step_size = step_size_adapter.finalize_step_size();
  }
}

std::vector<Eigen::VectorXd> NutsProposer::leapfrog(
    GlobalState& state,
    Eigen::VectorXd position,
    Eigen::VectorXd momentum,
    double direction) {
  // momentum half-step
  state.set_flattened_unconstrained_values(position);
  Eigen::VectorXd grad_U = compute_potential_gradient(state);
  momentum = momentum - direction * step_size * grad_U / 2;
  // position full-step
  position = position + direction * step_size * momentum;
  // momentum half-step
  state.set_flattened_unconstrained_values(position);
  grad_U = compute_potential_gradient(state);
  momentum = momentum - direction * step_size * grad_U / 2;

  return {position, momentum};
}

// Follows Algorithm 4 of NUTS paper
void NutsProposer::find_reasonable_step_size(
    GlobalState& state,
    std::mt19937& gen,
    Eigen::VectorXd position) {
  step_size = 1.0;
  Eigen::VectorXd momentum = initialize_momentum(position, gen);
  double acceptance_log_prob =
      compute_new_step_acceptance_probability(state, position, momentum);
  int a = 1;
  if (std::isnan(acceptance_log_prob) or acceptance_log_prob < std::log(0.5)) {
    a = -1;
  }
  const double LOG_2 = std::log(2.0);
  for (int i = 0; i < 100; i++) {
    double prev_step_size = step_size;
    step_size = std::pow(2, a) * step_size;
    acceptance_log_prob =
        compute_new_step_acceptance_probability(state, position, momentum);

    // don't increase step_size if acceptance is NaN
    if (std::isnan(acceptance_log_prob) and a > 1) {
      step_size = prev_step_size;
      break;
    }

    if (a * acceptance_log_prob <= -(LOG_2 * a)) {
      break;
    }
  }
}

double NutsProposer::compute_new_step_acceptance_probability(
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

double NutsProposer::compute_hamiltonian(
    GlobalState& state,
    Eigen::VectorXd position,
    Eigen::VectorXd momentum) {
  double K = compute_kinetic_energy(momentum);
  state.set_flattened_unconstrained_values(position);
  state.update_log_prob();
  double U = -state.get_log_prob();
  return K + U;
}

bool NutsProposer::compute_no_turn(
    Eigen::VectorXd position_left,
    Eigen::VectorXd momentum_left,
    Eigen::VectorXd position_right,
    Eigen::VectorXd momentum_right) {
  return (
      ((position_right - position_left).dot(momentum_left) >= 0.0) and
      ((position_right - position_left).dot(momentum_right) >= 0.0));
}

NutsProposer::Tree NutsProposer::build_tree_base_case(
    GlobalState& state,
    Eigen::VectorXd position,
    Eigen::VectorXd momentum,
    double slice,
    double direction,
    double hamiltonian_init) {
  Tree tree = Tree();

  std::vector<Eigen::VectorXd> leapfrog_result =
      leapfrog(state, position, momentum, direction);
  tree.position_new = leapfrog_result[0];
  Eigen::VectorXd momentum_new = leapfrog_result[1];

  tree.position_left = tree.position_new;
  tree.momentum_left = momentum_new;
  tree.position_right = tree.position_new;
  tree.momentum_right = momentum_new;
  tree.total_nodes = 1.0;

  double hamiltonian_new =
      compute_hamiltonian(state, tree.position_new, momentum_new);
  if (std::isnan(hamiltonian_new)) {
    tree.valid_nodes = 0.0;
    tree.no_turn = false;
    tree.acceptance_sum = 0.0;
    return tree;
  }

  tree.valid_nodes = slice <= -hamiltonian_new;
  // check for divergence
  tree.no_turn = slice < (delta_max - hamiltonian_new);

  double hamiltonian_diff = hamiltonian_init - hamiltonian_new;
  if (std::isnan(hamiltonian_diff)) {
    tree.acceptance_sum = 0.0;
  } else if (hamiltonian_diff > 0) {
    tree.acceptance_sum = 1.0;
  } else {
    tree.acceptance_sum = std::exp(hamiltonian_diff);
  }

  return tree;
}

NutsProposer::Tree NutsProposer::build_tree(
    GlobalState& state,
    std::mt19937& gen,
    Eigen::VectorXd position,
    Eigen::VectorXd momentum,
    double slice,
    double direction,
    int tree_depth,
    double hamiltonian_init) {
  if (tree_depth == 0) {
    return build_tree_base_case(
        state, position, momentum, slice, direction, hamiltonian_init);
  } else {
    Tree subtree1 = build_tree(
        state,
        gen,
        position,
        momentum,
        slice,
        direction,
        tree_depth - 1,
        hamiltonian_init);
    if (!subtree1.no_turn) {
      return subtree1;
    } else {
      Tree tree = Tree();
      tree.position_new = subtree1.position_new;

      Tree subtree2;
      if (direction < 0) {
        subtree2 = build_tree(
            state,
            gen,
            subtree1.position_left,
            subtree1.momentum_left,
            slice,
            direction,
            tree_depth - 1,
            hamiltonian_init);
        tree.position_left = subtree2.position_left;
        tree.momentum_left = subtree2.momentum_left;
        tree.position_right = subtree1.position_right;
        tree.momentum_right = subtree1.momentum_right;
      } else {
        subtree2 = build_tree(
            state,
            gen,
            subtree1.position_right,
            subtree1.momentum_right,
            slice,
            direction,
            tree_depth - 1,
            hamiltonian_init);
        tree.position_left = subtree1.position_left;
        tree.momentum_left = subtree1.momentum_left;
        tree.position_right = subtree2.position_right;
        tree.momentum_right = subtree2.momentum_right;
      }

      double update_prob = subtree2.valid_nodes /
          std::max(1.0, subtree1.valid_nodes + subtree2.valid_nodes);
      std::bernoulli_distribution update_dist(update_prob);
      if (update_dist(gen)) {
        tree.position_new = subtree2.position_new;
      }

      tree.acceptance_sum = subtree1.acceptance_sum + subtree2.acceptance_sum;
      tree.total_nodes = subtree1.total_nodes + subtree2.total_nodes;
      tree.no_turn = subtree2.no_turn and
          compute_no_turn(
                         tree.position_left,
                         tree.momentum_left,
                         tree.position_right,
                         tree.momentum_right);
      tree.valid_nodes = subtree1.valid_nodes + subtree2.valid_nodes;

      return tree;
    }
  }
}

// Follows Algorithm 6 of NUTS paper
double NutsProposer::propose(GlobalState& state, std::mt19937& gen) {
  Eigen::VectorXd position;
  state.get_flattened_unconstrained_values(position);

  // sample momentum
  Eigen::VectorXd momentum_init = initialize_momentum(position, gen);
  // sample slice
  std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
  double hamiltonian_init = compute_hamiltonian(state, position, momentum_init);
  double slice = std::log(uniform_dist(gen)) - hamiltonian_init;

  Eigen::VectorXd position_left = position;
  Eigen::VectorXd position_right = position;
  Eigen::VectorXd momentum_left = momentum_init;
  Eigen::VectorXd momentum_right = momentum_init;

  double valid_nodes = 1;
  double acceptance_sum = 0.0;
  double total_nodes = 0.0;

  std::bernoulli_distribution coin_flip(0.5);

  for (int tree_depth = 0; tree_depth < max_tree_depth; tree_depth++) {
    // sample direction
    double direction = -1.0;
    if (coin_flip(gen)) {
      direction = 1.0;
    }

    Tree tree;
    if (direction < 0) {
      tree = build_tree(
          state,
          gen,
          position_left,
          momentum_left,
          slice,
          direction,
          tree_depth,
          hamiltonian_init);
      position_left = tree.position_left;
      momentum_left = tree.momentum_left;
    } else {
      tree = build_tree(
          state,
          gen,
          position_right,
          momentum_right,
          slice,
          direction,
          tree_depth,
          hamiltonian_init);
      position_right = tree.position_right;
      momentum_right = tree.momentum_right;
    }

    acceptance_sum = tree.acceptance_sum;
    total_nodes = tree.total_nodes;
    if (!tree.no_turn) {
      break;
    }

    double update_prob = std::min(1.0, tree.valid_nodes / valid_nodes);
    std::bernoulli_distribution update_dist(update_prob);
    if (update_dist(gen)) {
      position = tree.position_new;
    }
    valid_nodes += tree.valid_nodes;

    bool no_turn = compute_no_turn(
        position_left, momentum_left, position_right, momentum_right);
    if (!no_turn) {
      break;
    }
  }

  warmup_acceptance_prob = acceptance_sum / total_nodes;

  state.set_flattened_unconstrained_values(position);
  state.update_log_prob();

  return 0.0;
}

} // namespace graph
} // namespace beanmachine
