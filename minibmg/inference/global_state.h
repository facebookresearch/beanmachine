/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
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
  // Create a global state that uses brute-force evaluation over the graph
  static std::unique_ptr<MinibmgGlobalState> create0(
      const beanmachine::minibmg::Graph& graph);

  // Create a global state that first compiles the model to an expression tree
  // and evaluates by interpreting that tree.
  static std::unique_ptr<MinibmgGlobalState> create1(
      const beanmachine::minibmg::Graph& graph);

  // Create a global state that first compiles the model to an expression tree,
  // generates code from that tree, and evaluates by running the generated code.
  static std::unique_ptr<MinibmgGlobalState> create2(
      const beanmachine::minibmg::Graph& graph);

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
  explicit MinibmgGlobalState(
      const beanmachine::minibmg::Graph& graph,
      std::unique_ptr<const HMCWorld> world);

  const beanmachine::minibmg::Graph& graph;
  const std::unique_ptr<const HMCWorld> world;
  std::vector<std::vector<beanmachine::graph::NodeValue>> samples;
  int flat_size;
  double log_prob;
  std::vector<double> unconstrained_values;
  std::vector<double> unconstrained_grads;
  std::vector<double> saved_unconstrained_values;
  std::vector<double> saved_unconstrained_grads;
};

} // namespace beanmachine::minibmg
