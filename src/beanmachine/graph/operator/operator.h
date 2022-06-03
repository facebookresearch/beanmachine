/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <map>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace oper {

class Operator : public graph::Node {
 public:
  explicit Operator(graph::OperatorType op_type)
      : graph::Node(graph::NodeType::OPERATOR), op_type(op_type) {}
  ~Operator() override {}
  bool is_stochastic() const override {
    return false;
  }
  double log_prob() const override;
  void eval(std::mt19937& gen) override;

  // Computes gradients of node's value based on its in-nodes values and
  // gradients (forward autodifferentiation).
  // TODO: eval_gradient would be a better name, as it mirrors eval() very well,
  // but since a general refactoring in planned (as of May 2022),
  // we will wait until then.
  void compute_gradients() override;

  // Computes the gradient of the log probability of this node's value based on
  // its in-nodes values and gradients (forward autodifferentiation). It is
  // conceptually very similar to compute_gradients, but we cannot use
  // compute_gradients for the log probability because it is not a node.
  // Because only stochastic operators contribute to log prob,
  // this method is only implemented for stochastic operators.
  // TODO: should we just remove it from Operator then?
  void gradient_log_prob(
      const graph::Node* target_node,
      double& grad1,
      double& grad2) const override;

  void backward() override {}
  graph::OperatorType op_type;
};

class OperatorFactory {
 public:
  typedef std::unique_ptr<Operator> (*builder_type)(
      const std::vector<graph::Node*>&);

  OperatorFactory() = delete;
  ~OperatorFactory() {}

  static bool register_op(
      const graph::OperatorType op_type,
      builder_type op_builder);
  static std::unique_ptr<Operator> create_op(
      const graph::OperatorType op_type,
      const std::vector<graph::Node*>& in_nodes);

 private:
  static std::map<int, builder_type>& op_map();
};

} // namespace oper
} // namespace beanmachine
