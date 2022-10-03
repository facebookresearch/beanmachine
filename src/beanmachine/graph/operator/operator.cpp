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

using namespace std;

unique_ptr<graph::Node> Operator::clone() {
  return OperatorFactory::create_op(op_type, in_nodes);
}

double Operator::log_prob() const {
  throw runtime_error("log_prob is only defined for sample or iid sample");
}

void Operator::gradient_log_prob(
    const graph::Node* target_node,
    double& /* grad1 */,
    double& /* grad2 */) const {
  throw runtime_error(
      "gradient_log_prob is only defined for sample or iid sample");
}

void Operator::eval(mt19937& /* gen */) {
  throw runtime_error(
      "internal error: unexpected operator type " +
      std::to_string(static_cast<int>(op_type)) + " at node_id " +
      std::to_string(index));
}

void Operator::compute_gradients() {
  throw runtime_error(
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

unique_ptr<Operator> OperatorFactory::create_op(
    const graph::OperatorType op_type,
    const vector<graph::Node*>& in_nodes) {
  int op_id = static_cast<int>(op_type);
  // Check OperatorFactory::factories_are_registered here to deactivate compiler
  // optimization on unused static is_registered variables.
  if (!OperatorFactory::factories_are_registered) {
    throw runtime_error(
        "internal error: unregistered operator type " + to_string(op_id));
  }
  auto iter = OperatorFactory::op_map().find(op_id);
  if (iter != OperatorFactory::op_map().end()) {
    return iter->second(in_nodes);
  }
  throw runtime_error(
      "internal error: unregistered operator type " + to_string(op_id));
  return nullptr;
}

map<int, OperatorFactory::builder_type>& OperatorFactory::op_map() {
  static map<int, OperatorFactory::builder_type> operator_map;
  return operator_map;
}

} // namespace oper
} // namespace beanmachine
