// Copyright (c) Facebook, Inc. and its affiliates.

#include "beanmachine/graph/operator/controlop.h"
#include "beanmachine/graph/operator/multiaryop.h"
#include "beanmachine/graph/operator/operator.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/operator/unaryop.h"

namespace beanmachine {
namespace oper {

double Operator::log_prob() const {
  throw std::runtime_error("log_prob is only defined for sample or iid sample");
}

void Operator::gradient_log_prob(double& /* grad1 */, double& /* grad2 */) const {
  throw std::runtime_error(
      "gradient_log_prob is only defined for sample or iid sample");
}

void Operator::gradient_log_prob(
    Eigen::MatrixXd& /* grad1 */,
    Eigen::MatrixXd& /* grad2_diag */) const {
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
  if (iter != OperatorFactory::op_map().end()) {
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
