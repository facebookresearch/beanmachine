/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/proposer/global_proposer.h"
#include "beanmachine/graph/global/proposer/hmc_util.h"

namespace beanmachine {
namespace graph {

class HmcProposer : public GlobalProposer {
 public:
  explicit HmcProposer(
      double path_length,
      double step_size = 0.1,
      bool adapt_mass_matrix = true,
      double optimal_acceptance_prob = 0.65);
  void initialize(GlobalState& state, std::mt19937& gen, int num_warmup_samples)
      override;
  void warmup(
      GlobalState& state,
      std::mt19937& gen,
      double acceptance_log_prob,
      int iteration,
      int num_warmup_samples) override;
  double propose(GlobalState& state, std::mt19937& gen) override;

 protected:
  StepSizeAdapter step_size_adapter;
  WindowedMassMatrixAdapter mass_matrix_adapter;
  double path_length;
  double step_size;
  bool adapt_mass_matrix;
  int max_steps;
  Eigen::MatrixXd mass_inv;
  Eigen::ArrayXd mass_matrix_diagonal;
  double compute_kinetic_energy(Eigen::VectorXd momentum);
  Eigen::VectorXd compute_potential_gradient(GlobalState& state);
  Eigen::VectorXd initialize_momentum(Eigen::VectorXd theta, std::mt19937& gen);
  void find_reasonable_step_size(
      GlobalState& state,
      std::mt19937& gen,
      Eigen::VectorXd position);
  double compute_new_step_acceptance_probability(
      GlobalState& state,
      Eigen::VectorXd position,
      Eigen::VectorXd momentum);
  double compute_hamiltonian(
      GlobalState& state,
      Eigen::VectorXd theta,
      Eigen::VectorXd r);
  std::vector<Eigen::VectorXd> leapfrog(
      GlobalState& state,
      Eigen::VectorXd theta,
      Eigen::VectorXd r,
      double v);
};

} // namespace graph
} // namespace beanmachine
