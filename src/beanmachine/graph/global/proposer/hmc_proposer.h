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
      double optimal_acceptance_prob = 0.65);
  void initialize(GlobalState& state, std::mt19937& gen, int num_warmup_samples)
      override;
  void warmup(double acceptance_log_prob, int iteration, int num_warmup_samples)
      override;
  double propose(GlobalState& state, std::mt19937& gen) override;

 protected:
  StepSizeAdapter step_size_adapter;
  double path_length;
  double step_size;
  double compute_kinetic_energy(Eigen::VectorXd momentum);
  Eigen::VectorXd compute_potential_gradient(GlobalState& state);
  Eigen::VectorXd initialize_momentum(Eigen::VectorXd theta, std::mt19937& gen);
};

} // namespace graph
} // namespace beanmachine
