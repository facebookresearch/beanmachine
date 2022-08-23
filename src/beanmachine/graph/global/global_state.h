/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

enum class InitType { RANDOM, ZERO, PRIOR };

class GlobalState {
 public:
  explicit GlobalState(Graph& g);
  void initialize_values(InitType init_type, uint seed);
  void backup_unconstrained_values();
  void backup_unconstrained_grads();
  void revert_unconstrained_values();
  void revert_unconstrained_grads();
  void add_to_stochastic_unconstrained_nodes(Eigen::VectorXd& increment);
  void get_flattened_unconstrained_values(Eigen::VectorXd& flattened_values);
  void set_flattened_unconstrained_values(Eigen::VectorXd& flattened_values);
  void get_flattened_unconstrained_grads(Eigen::VectorXd& flattened_grad);
  double get_log_prob();
  void update_log_prob();
  void update_backgrad();

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
