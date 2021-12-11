/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace oper {

void StochasticOperator::gradient_log_prob(
    const graph::Node* target_node,
    double& log_prob_grad1,
    double& log_prob_grad2) const {
  // The implementation of this method is a little subtle.
  //
  // We want to compute the first and second gradients
  // of log prob
  // with respect to target_node,
  // given the gradients of
  // the inputs to that function,
  // also with respect to target_node,
  // stored in these inputs' fields grad1 and grad2.
  //
  // The standard way of computing that is just to use the chain rule.
  // This is straightforward for deterministic nodes because
  // the function inputs correspond exactly to the in-nodes,
  // and the function for which the derivative is being computed is
  // the deterministic node itself.
  // So the chain rule applies pretty directly and straightforwardly.
  //
  // For stochastic nodes we have two complications beyond that.
  // * we are not computing the derivative of this stochastic node's value,
  //   we are computing the derivative of the log prob,
  //   which is in fact not represented by *any* node.
  // * The in-nodes of the stochastic node are *not* *all*
  //   the inputs of the log prob function.
  //   The value of the stochastic node is an additional input to log prob,
  //   but it is not represented as an in-node.
  //
  // This makes the computation of this derivative less uniform
  // and less directly corresponding to the typical use of the chain rule.
  //
  // Still, it should be possible to simply apply the chain rule
  // and find an expression involving the gradient of this stochastic node's
  // value and the gradients of the other in-nodes.
  // The only disadvantage to that is that the treatment of
  // log prob's inputs would not be perfectly uniform,
  // with the stochastic operator's value being treated differently.
  //
  // At this point, we note something else.
  // We are computing the gradients with respect to (possibly distant)
  // target_node. Let's denote target_node by simply t. For v the value of the
  // current (this) stochastic node and p the remaining inputs (the in-nodes,
  // or more precisely the parameters of the distribution from which v was
  // sampled), we have:
  // d(log prob(v,p))/dt =
  //     d(log prob(v,p))/dv * dv/dt   +
  //     d(log prob(v,p))/dp * dp/dt.
  //
  // Note that the value v is not determined by any other nodes in the model,
  // because it is sampled (it has no ancestors).
  //
  // Therefore, the only way dv/dt is not 0 is if t is v itself.
  // In that case, dp/dt will be zero because v is not an ancestor of p.
  // If t is *not* v, then dv/dt is zero
  // (for t is not an ancestor of v since no node is).
  //
  // Therefore, at least one of dv/dt or dp/dt will be zero,
  // so the final gradient is dv/dt if t is v, and is dp/dt otherwise.

  const auto dist = static_cast<const distribution::Distribution*>(in_nodes[0]);
  if (this == target_node) {
    dist->gradient_log_prob_value(value, log_prob_grad1, log_prob_grad2);
  } else {
    dist->gradient_log_prob_param(value, log_prob_grad1, log_prob_grad2);
  }
}

graph::NodeValue* StochasticOperator::get_original_value(
    bool sync_from_unconstrained) {
  if (transform_type != graph::TransformType::NONE and
      sync_from_unconstrained) {
    transform->inverse(value, unconstrained_value);
  }
  return &value;
}

graph::NodeValue* StochasticOperator::get_unconstrained_value(
    bool sync_from_constrained) {
  if (transform_type == graph::TransformType::NONE) {
    return &value;
  }
  if (unconstrained_value.type == graph::AtomicType::UNKNOWN) {
    unconstrained_value.type = graph::ValueType(
        value.type.variable_type,
        graph::AtomicType::REAL,
        value.type.rows,
        value.type.cols);
  }
  if (sync_from_constrained) {
    transform->operator()(value, unconstrained_value);
  }
  return &unconstrained_value;
}

double StochasticOperator::log_abs_jacobian_determinant() {
  if (transform_type == graph::TransformType::NONE) {
    return 0;
  }
  return transform->log_abs_jacobian_determinant(value, unconstrained_value);
}

graph::DoubleMatrix* StochasticOperator::get_unconstrained_gradient() {
  if (transform_type != graph::TransformType::NONE) {
    transform->unconstrained_gradient(back_grad1, value, unconstrained_value);
  }
  return &back_grad1;
}

void Sample::_backward(bool skip_observed) {
  const auto dist = static_cast<distribution::Distribution*>(in_nodes[0]);
  dist->backward_param(value);
  if (!(is_observed and skip_observed)) {
    dist->backward_value(value, back_grad1);
  }
}

void IIdSample::_backward(bool skip_observed) {
  const auto dist = static_cast<distribution::Distribution*>(in_nodes[0]);
  dist->backward_param_iid(value);
  if (!(is_observed and skip_observed)) {
    dist->backward_value_iid(value, back_grad1);
  }
}

Sample::Sample(const std::vector<graph::Node*>& in_nodes)
    : StochasticOperator(graph::OperatorType::SAMPLE) {
  if (in_nodes.size() != 1) {
    throw std::invalid_argument("operator SAMPLE requires a single parent");
  }
  if (in_nodes[0]->node_type != graph::NodeType::DISTRIBUTION) {
    throw std::invalid_argument(
        "operator SAMPLE requires a distribution parent");
  }
  const distribution::Distribution* dist =
      static_cast<distribution::Distribution*>(in_nodes[0]);
  // The type of value of a SAMPLE node is obviously the sample type
  // of the distribution parent.
  value = graph::NodeValue(dist->sample_type);
  // For the unconstrained value we want to avoid unnecessary early memory
  // allocation; just set it to a real scalar or 1x0 array of reals.
  auto vt = dist->sample_type.variable_type;
  if (vt == graph::VariableType::COL_SIMPLEX_MATRIX) {
    vt = graph::VariableType::BROADCAST_MATRIX;
  }
  auto at = graph::AtomicType::REAL;
  unconstrained_value = graph::NodeValue(graph::ValueType(vt, at, 1, 0));
}

IIdSample::IIdSample(const std::vector<graph::Node*>& in_nodes)
    : StochasticOperator(graph::OperatorType::IID_SAMPLE) {
  uint in_degree = static_cast<uint>(in_nodes.size());
  if (in_degree != 2 and in_degree != 3) {
    throw std::invalid_argument("operator IID_SAMPLE requires 2 or 3 parents");
  }
  if (in_nodes[0]->node_type != graph::NodeType::DISTRIBUTION) {
    throw std::invalid_argument(
        "operator IID_SAMPLE requires the first parent to be a distribution");
  }
  if (in_nodes[1]->value.type != graph::AtomicType::NATURAL) {
    throw std::invalid_argument(
        "operator IID_SAMPLE requires the second parent to be NATURAL");
  }
  if (in_nodes[1]->node_type != graph::NodeType::CONSTANT) {
    throw std::invalid_argument(
        "operator IID_SAMPLE requires the second parent to be CONSTANT");
  }

  if (in_degree == 3) {
    if (in_nodes[2]->value.type != graph::AtomicType::NATURAL) {
      throw std::invalid_argument(
          "operator IID_SAMPLE requires the third parent to be NATURAL");
    }

    if (in_nodes[2]->node_type != graph::NodeType::CONSTANT) {
      throw std::invalid_argument(
          "operator IID_SAMPLE requires the third parent to be CONSTANT");
    }
  }
  // CONSIDER removing this restriction. Sure, it is somewhat pointless to
  // create a one-value matrix here, but if the caller has a reason to create a
  // one-value matrix, then why not let them?
  if (in_degree == 2 and in_nodes[1]->value._natural < 2) {
    throw std::invalid_argument(
        "operator IID_SAMPLE requires the second or third parent to have value >= 2");
  }
  if (in_degree == 3 and
      (in_nodes[1]->value._natural * in_nodes[2]->value._natural) < 2) {
    throw std::invalid_argument(
        "operator IID_SAMPLE requires the second or third parent to have value >= 2");
  }
  const auto dist = static_cast<distribution::Distribution*>(in_nodes[0]);

  // determine the value type
  graph::ValueType vtype;
  switch (dist->sample_type.variable_type) {
    case graph::VariableType::SCALAR:
      vtype = graph::ValueType(
          graph::VariableType::BROADCAST_MATRIX,
          dist->sample_type.atomic_type,
          static_cast<uint>(in_nodes[1]->value._natural),
          in_degree == 2 ? 1 : static_cast<uint>(in_nodes[2]->value._natural));
      break;
    case graph::VariableType::BROADCAST_MATRIX:
    case graph::VariableType::COL_SIMPLEX_MATRIX:
    default:
      throw std::invalid_argument(
          "operator IID_SAMPLE requires the sample type to be a non-matrix type");
  }
  value = graph::NodeValue(vtype);
  // leave uninitialized until necessary
  unconstrained_value = graph::NodeValue(graph::AtomicType::UNKNOWN);
  return;
}

} // namespace oper
} // namespace beanmachine
