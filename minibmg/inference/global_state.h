/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "beanmachine/graph/global/global_state.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/ad/reverse.h"
#include "beanmachine/minibmg/graph.h"
#include "hmc_world.h"

namespace beanmachine::minibmg {

// using namespace beanmachine::graph;

// Global state, an implementation of beanmachine::graph::GlobalState which is
// needed to use the NUTS api from bmg.
class MinibmgGlobalState : public beanmachine::graph::GlobalState {
 public:
  explicit MinibmgGlobalState(beanmachine::minibmg::Graph& graph);
  void initialize_values(beanmachine::graph::InitType init_type, uint seed)
      override;
  void backup_unconstrained_values() override;
  void backup_unconstrained_grads() override;
  void revert_unconstrained_values() override;
  void revert_unconstrained_grads() override;
  void add_to_stochastic_unconstrained_nodes(
      Eigen::VectorXd& increment) override;
  void get_flattened_unconstrained_values(
      Eigen::VectorXd& flattened_values) override;
  void set_flattened_unconstrained_values(
      Eigen::VectorXd& flattened_values) override;
  void get_flattened_unconstrained_grads(
      Eigen::VectorXd& flattened_grad) override;
  double get_log_prob() override;
  void update_log_prob() override;
  void update_backgrad() override;
  void collect_sample() override;
  std::vector<std::vector<beanmachine::graph::NodeValue>>& get_samples()
      override;
  void set_default_transforms() override;
  void set_agg_type(beanmachine::graph::AggregationType) override;
  void clear_samples() override;

 private:
  const beanmachine::minibmg::Graph& graph;
  const std::unique_ptr<const HMCWorld> world;
  std::vector<std::vector<beanmachine::graph::NodeValue>> samples;
  int flat_size;
  double log_prob;
  std::vector<double> unconstrained_values;
  std::vector<double> unconstrained_grads;
  std::vector<double> saved_unconstrained_values;
  std::vector<double> saved_unconstrained_grads;

  // scratchpads for evaluation
  std::unordered_map<Nodep, Reverse<Real>> reverse_eval_data;
  std::unordered_map<Nodep, Real> real_eval_data;
};

} // namespace beanmachine::minibmg
