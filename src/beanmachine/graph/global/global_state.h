/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

enum class InitType { RANDOM, ZERO, PRIOR };

class GlobalState {
 public:
  virtual void initialize_values(InitType init_type, uint seed) = 0;
  virtual void backup_unconstrained_values() = 0;
  virtual void backup_unconstrained_grads() = 0;
  virtual void revert_unconstrained_values() = 0;
  virtual void revert_unconstrained_grads() = 0;
  virtual void add_to_stochastic_unconstrained_nodes(
      Eigen::VectorXd& increment) = 0;
  virtual void get_flattened_unconstrained_values(
      Eigen::VectorXd& flattened_values) = 0;
  virtual void set_flattened_unconstrained_values(
      Eigen::VectorXd& flattened_values) = 0;
  virtual void get_flattened_unconstrained_grads(
      Eigen::VectorXd& flattened_grad) = 0;
  virtual double get_log_prob() = 0;
  virtual void update_log_prob() = 0;
  virtual void update_backgrad() = 0;
  virtual void collect_sample() = 0;
  virtual std::vector<std::vector<NodeValue>>& get_samples() = 0;
  virtual void set_default_transforms() = 0;
  virtual void set_agg_type(AggregationType) = 0;
  virtual void clear_samples() = 0;

  virtual ~GlobalState() {}
};

class GraphGlobalState : public GlobalState {
 public:
  explicit GraphGlobalState(Graph& g);
  void initialize_values(InitType init_type, uint seed) override;
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
  std::vector<std::vector<NodeValue>>& get_samples() override;
  void set_default_transforms() override;
  void set_agg_type(AggregationType) override;
  void clear_samples() override;

 private:
  int flat_size;
  Graph& graph;
  std::vector<Node*> stochastic_nodes;
  std::vector<Node*> deterministic_nodes;
  std::vector<NodeValue> stochastic_unconstrained_vals_backup;
  std::vector<DoubleMatrix> stochastic_unconstrained_grads_backup;
  double log_prob;
};

} // namespace graph
} // namespace beanmachine
