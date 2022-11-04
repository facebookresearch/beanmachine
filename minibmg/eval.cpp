/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/eval.h"
#include <stdexcept>

namespace beanmachine::minibmg {

RecursiveNodeEvaluatorVisitor::RecursiveNodeEvaluatorVisitor(
    std::function<double(const std::string& name, const int identifier)>
        read_variable)
    : read_variable{read_variable} {}

void RecursiveNodeEvaluatorVisitor::visit(const ScalarVariableNode* node) {
  result = read_variable(node->name, node->identifier);
}

void RecursiveNodeEvaluatorVisitor::visit(const ScalarSampleNode*) {
  throw std::logic_error("recursive evaluator may not sample");
}

Real RecursiveNodeEvaluatorVisitor::evaluate_input(const ScalarNodep& node) {
  return evaluate_scalar(node);
}

std::shared_ptr<const Distribution<Real>>
RecursiveNodeEvaluatorVisitor::evaluate_input_distribution(
    const DistributionNodep&) {
  throw std::logic_error(
      "recursive evaluator may not traffic in distributions");
}

double eval_node(
    RecursiveNodeEvaluatorVisitor& evaluator,
    const ScalarNodep& node) {
  return evaluator.evaluate_scalar(node).value;
}

} // namespace beanmachine::minibmg
