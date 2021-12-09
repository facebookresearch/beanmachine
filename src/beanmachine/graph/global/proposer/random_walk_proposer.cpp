/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/proposer/random_walk_proposer.h"

namespace beanmachine {
namespace graph {

RandomWalkProposer::RandomWalkProposer(double step_size) : GlobalProposer() {
  this->step_size = step_size;
}

double RandomWalkProposer::propose(GlobalState& state, std::mt19937& gen) {
  double initial_log_prob = state.get_log_prob();

  Eigen::VectorXd flattened_values;
  state.get_flattened_unconstrained_values(flattened_values);

  std::normal_distribution<double> dist(0.0, 1.0);
  for (int i = 0; i < flattened_values.size(); i++) {
    flattened_values[i] += step_size * dist(gen);
  }
  state.set_flattened_unconstrained_values(flattened_values);
  state.update_log_prob();
  double final_log_prob = state.get_log_prob();

  return final_log_prob - initial_log_prob;
}

} // namespace graph
} // namespace beanmachine
