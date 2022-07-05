/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/proposer/hmc_proposer.h"

namespace beanmachine {
namespace graph {

class NutsProposer : public HmcProposer {
 public:
  explicit NutsProposer(
      bool adapt_mass_matrix = true,
      bool multinomial_sampling = true,
      double optimal_acceptance_prob = 0.8);
  void initialize(GlobalState& state, std::mt19937& gen, int num_warmup_samples)
      override;
  void warmup(
      GlobalState& state,
      std::mt19937& gen,
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
    Eigen::VectorXd momentum_sum;
    double log_weight;
    bool no_turn;
    double acceptance_sum;
    double total_nodes;
  };
  bool multinomial_sampling;
  double warmup_acceptance_prob;
  double delta_max;
  double max_tree_depth;
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
  bool compute_no_turn(
      Eigen::VectorXd momentum_left,
      Eigen::VectorXd momentum_right,
      Eigen::VectorXd momentum_sum);
};

} // namespace graph
} // namespace beanmachine
