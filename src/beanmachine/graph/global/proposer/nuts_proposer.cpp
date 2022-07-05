/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/proposer/nuts_proposer.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace graph {

NutsProposer::NutsProposer(
    bool adapt_mass_matrix,
    bool multinomial_sampling,
    double optimal_acceptance_prob)
    : HmcProposer(0.0, 1.0, adapt_mass_matrix, optimal_acceptance_prob) {
  this->multinomial_sampling = multinomial_sampling;
  step_size = 1.0; // will be updated in `find_reasonable_step_size`
  delta_max = 1000;
  max_tree_depth = 10;
}

void NutsProposer::initialize(
    GlobalState& state,
    std::mt19937& gen,
    int num_warmup_samples) {
  Eigen::VectorXd position;
  state.get_flattened_unconstrained_values(position);

  int size = static_cast<int>(position.size());
  mass_inv = Eigen::MatrixXd::Identity(size, size);
  mass_matrix_diagonal = Eigen::ArrayXd::Ones(size);
  if (adapt_mass_matrix) {
    mass_matrix_adapter.initialize(num_warmup_samples, size);
  }

  step_size_adapter.initialize(step_size);
  find_reasonable_step_size(state, gen, position);
}

void NutsProposer::warmup(
    GlobalState& state,
    std::mt19937& gen,
    double /*acceptance_prob*/,
    int iteration,
    int num_warmup_samples) {
  step_size =
      step_size_adapter.update_step_size(iteration, warmup_acceptance_prob);

  if (adapt_mass_matrix) {
    Eigen::VectorXd sample;
    state.get_flattened_unconstrained_values(sample);
    mass_matrix_adapter.update_mass_matrix(iteration, sample);
    bool window_end = mass_matrix_adapter.is_end_window(iteration);

    if (window_end) {
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

bool NutsProposer::compute_no_turn(
    Eigen::VectorXd momentum_left,
    Eigen::VectorXd momentum_right,
    Eigen::VectorXd momentum_sum) {
  Eigen::VectorXd transformed_right =
      mass_inv.diagonal().array() * momentum_right.array();
  Eigen::VectorXd transformed_left =
      mass_inv.diagonal().array() * momentum_left.array();
  return (transformed_right.dot(momentum_sum) >= 0.0) and
      (transformed_left.dot(momentum_sum) >= 0.0);
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
  tree.momentum_sum = momentum_new;
  tree.total_nodes = 1.0;

  double hamiltonian_new =
      compute_hamiltonian(state, tree.position_new, momentum_new);
  if (std::isnan(hamiltonian_new)) {
    tree.log_weight = -std::numeric_limits<double>::infinity();
    tree.no_turn = false;
    tree.acceptance_sum = 0.0;
    return tree;
  }

  double hamiltonian_diff = hamiltonian_init - hamiltonian_new;
  if (hamiltonian_diff > 0) {
    tree.acceptance_sum = 1.0;
  } else {
    tree.acceptance_sum = std::exp(hamiltonian_diff);
  }

  if (multinomial_sampling) {
    tree.log_weight = hamiltonian_diff;
  } else {
    tree.log_weight = std::log(slice <= -hamiltonian_new);
  }
  // check for divergence
  tree.no_turn = slice < (delta_max - hamiltonian_new);

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
      Tree tree = subtree1;
      tree.momentum_sum = Eigen::VectorXd::Zero(subtree1.momentum_sum.size());
      tree.log_weight = -std::numeric_limits<double>::infinity();
      return tree;
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

      tree.total_nodes = subtree1.total_nodes + subtree2.total_nodes;
      tree.acceptance_sum = subtree1.acceptance_sum + subtree2.acceptance_sum;
      if (!subtree2.no_turn) {
        tree.momentum_sum = Eigen::VectorXd::Zero(subtree1.momentum_sum.size());
        tree.log_weight = -std::numeric_limits<double>::infinity();
        return tree;
      }

      tree.log_weight =
          util::log_sum_exp(subtree1.log_weight, subtree2.log_weight);
      double update_log_prob = subtree2.log_weight - tree.log_weight;
      if (util::sample_logprob(gen, update_log_prob)) {
        tree.position_new = subtree2.position_new;
      }

      tree.momentum_sum = subtree1.momentum_sum + subtree2.momentum_sum;
      tree.no_turn =
          subtree2.no_turn and
          compute_no_turn(
              tree.momentum_left, tree.momentum_right, tree.momentum_sum);

      Tree left_tree;
      Tree right_tree;
      if (direction > 0) {
        left_tree = subtree1;
        right_tree = subtree2;
      } else {
        left_tree = subtree2;
        right_tree = subtree1;
      }
      tree.no_turn = tree.no_turn and
          compute_no_turn(
                         left_tree.momentum_left,
                         right_tree.momentum_left,
                         left_tree.momentum_sum + right_tree.momentum_left);
      tree.no_turn = tree.no_turn and
          compute_no_turn(
                         right_tree.momentum_right,
                         left_tree.momentum_right,
                         right_tree.momentum_sum + left_tree.momentum_right);

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
  double slice;
  if (multinomial_sampling) {
    slice = -hamiltonian_init;
  } else {
    slice = std::log(uniform_dist(gen)) - hamiltonian_init;
  }

  std::bernoulli_distribution coin_flip(0.5);

  Tree current_tree = {
      position,
      momentum_init,
      position,
      momentum_init,
      position,
      momentum_init,
      0.0,
      true,
      0.0,
      0.0};
  Tree left_tree;
  Tree right_tree;
  for (int tree_depth = 0; tree_depth < max_tree_depth; tree_depth++) {
    // sample direction
    double direction = -1.0;
    if (coin_flip(gen)) {
      direction = 1.0;
    }

    Tree new_tree;
    if (direction < 0) {
      // build tree to the left
      new_tree = build_tree(
          state,
          gen,
          current_tree.position_left,
          current_tree.momentum_left,
          slice,
          direction,
          tree_depth,
          hamiltonian_init);
      right_tree = current_tree;
      left_tree = new_tree;
    } else {
      // build tree to the right
      new_tree = build_tree(
          state,
          gen,
          current_tree.position_right,
          current_tree.momentum_right,
          slice,
          direction,
          tree_depth,
          hamiltonian_init);
      left_tree = current_tree;
      right_tree = new_tree;
    }

    current_tree.position_left = left_tree.position_left;
    current_tree.momentum_left = left_tree.momentum_left;
    current_tree.position_right = right_tree.position_right;
    current_tree.momentum_right = right_tree.momentum_right;

    current_tree.acceptance_sum += new_tree.acceptance_sum;
    current_tree.total_nodes += new_tree.total_nodes;
    if (!new_tree.no_turn) {
      break;
    }

    double update_log_prob = new_tree.log_weight - current_tree.log_weight;
    if (util::sample_logprob(gen, update_log_prob)) {
      current_tree.position_new = new_tree.position_new;
    }
    current_tree.log_weight =
        util::log_sum_exp(current_tree.log_weight, new_tree.log_weight);

    bool no_turn = new_tree.no_turn and
        compute_no_turn(
                       left_tree.momentum_left,
                       right_tree.momentum_right,
                       left_tree.momentum_sum + right_tree.momentum_sum);
    // check condition of left tree and leftmost node of right tree
    no_turn &= compute_no_turn(
        left_tree.momentum_left,
        right_tree.momentum_left,
        left_tree.momentum_sum + right_tree.momentum_left);
    // check condition of right tree and rightmost node of left tree
    no_turn &= compute_no_turn(
        right_tree.momentum_right,
        left_tree.momentum_right,
        right_tree.momentum_sum + left_tree.momentum_right);

    current_tree.momentum_sum += new_tree.momentum_sum;
    if (!no_turn) {
      break;
    }
  }

  warmup_acceptance_prob =
      current_tree.acceptance_sum / current_tree.total_nodes;

  state.set_flattened_unconstrained_values(current_tree.position_new);
  state.update_log_prob();

  return 0.0;
}

} // namespace graph
} // namespace beanmachine
