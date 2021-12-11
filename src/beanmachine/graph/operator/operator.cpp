/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/operator/operator.h"
#include "beanmachine/graph/operator/controlop.h"
#include "beanmachine/graph/operator/multiaryop.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/operator/unaryop.h"

namespace beanmachine {
namespace oper {

double Operator::log_prob() const {
  throw std::runtime_error("log_prob is only defined for sample or iid sample");
}

void Operator::gradient_log_prob(
    const graph::Node* target_node,
    double& /* grad1 */,
    double& /* grad2 */) const {
  throw std::runtime_error(
      "gradient_log_prob is only defined for sample or iid sample");
}

void Operator::eval(std::mt19937& /* gen */) {
  throw std::runtime_error(
      "internal error: unexpected operator type " +
      std::to_string(static_cast<int>(op_type)) + " at node_id " +
      std::to_string(index));
}

void Operator::compute_gradients() {
  throw std::runtime_error(
      "internal error: unexpected operator type " +
      std::to_string(static_cast<int>(op_type)) + " at node_id " +
      std::to_string(index));
}

bool OperatorFactory::register_op(
    const graph::OperatorType op_type,
    builder_type op_builder) {
  int op_id = static_cast<int>(op_type);
  auto iter = OperatorFactory::op_map().find(op_id);
  if (iter == OperatorFactory::op_map().end()) {
    OperatorFactory::op_map()[op_id] = op_builder;
    return true;
  }
  return false;
}

std::unique_ptr<Operator> OperatorFactory::create_op(
    const graph::OperatorType op_type,
    const std::vector<graph::Node*>& in_nodes) {
  int op_id = static_cast<int>(op_type);
  auto iter = OperatorFactory::op_map().find(op_id);
  // Check Sample::is_registered here to deactivate compiler optimization on
  // unused static is_registered variables.
  if (iter != OperatorFactory::op_map().end() and Sample::is_registered) {
    return iter->second(in_nodes);
  }
  throw std::runtime_error(
      "internal error: unregistered operator type " + std::to_string(op_id));
  return nullptr;
}

std::map<int, OperatorFactory::builder_type>& OperatorFactory::op_map() {
  static std::map<int, OperatorFactory::builder_type> operator_map;
  return operator_map;
}

} // namespace oper
} // namespace beanmachine
