/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/proposer/hmc_proposer.h"

namespace beanmachine {
namespace graph {

class NutsProposer : public HmcProposer {
 public:
  explicit NutsProposer(double optimal_acceptance_prob = 0.6);
  void initialize(
      GlobalState& state,
      std::mt19937& gen,
      int /* num_warmup_samples */) override;
  void warmup(
      double /* acceptance_log_prob */,
      int iteration,
      int num_warmup_samples) override;
  double propose(GlobalState& state, std::mt19937& gen) override;

 private:
  struct Tree {
    Eigen::VectorXd position_left;
    Eigen::VectorXd momentum_left;
    Eigen::VectorXd position_right;
    Eigen::VectorXd momentum_right;
    Eigen::VectorXd position_new;
    double valid_nodes;
    bool no_turn;
    double acceptance_sum;
    double total_nodes;
  };
  double step_size;
  double warmup_acceptance_prob;
  double delta_max;
  double max_tree_depth;
  void find_reasonable_step_size(
      GlobalState& state,
      std::mt19937& gen,
      Eigen::VectorXd position);
  double compute_new_step_acceptance_probability(
      GlobalState& state,
      Eigen::VectorXd position,
      Eigen::VectorXd momentum);
  std::vector<Eigen::VectorXd> leapfrog(
      GlobalState& state,
      Eigen::VectorXd theta,
      Eigen::VectorXd r,
      double v);
  Tree build_tree_base_case(
      GlobalState& state,
      Eigen::VectorXd position,
      Eigen::VectorXd momentum,
      double slice,
      double direction,
      double hamiltonian_init);
  Tree build_tree(
      GlobalState& state,
      std::mt19937& gen,
      Eigen::VectorXd position,
      Eigen::VectorXd momentum,
      double slice,
      double direction,
      int tree_depth,
      double hamiltonian_init);
  double compute_hamiltonian(
      GlobalState& state,
      Eigen::VectorXd theta,
      Eigen::VectorXd r);
  bool compute_no_turn(
      Eigen::VectorXd position_left,
      Eigen::VectorXd momentum_left,
      Eigen::VectorXd position_right,
      Eigen::VectorXd momentum_right);
};

} // namespace graph
} // namespace beanmachine
