// Copyright (c) Facebook, Inc. and its affiliates.

#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace oper {

void StochasticOperator::gradient_log_prob(
    double& log_prob_grad1,
    double& log_prob_grad2) const {
  // The implementation of this method is a little subtle.
  //
  // We want to compute the gradient of log prob given the gradient of
  // the inputs to that function.
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
  // We are computing the derivative with respect to some (possibly distant)
  // quantity z. For v the value of this stochastic node and
  // p the remaining inputs (the in-nodes), we have:
  // d(log prob(v,p))/dz =
  //     d(log prob(v,p))/dv * dv/dz   +
  //     d(log prob(v,p))/dp * dp/dz.
  //
  // Note that the value v is not determined by any other nodes in the model,
  // because it is sampled (it has no ancestors).
  //
  // Therefore, the only way dv/dz is not 0 is if z is v itself.
  // In that case, dp/dz will be zero because v is not an ancestor of p.
  // If z is *not* v, then dv/dz is zero
  // (for z is not an ancestor of v since no other node is).
  //
  // Therefore, at least one of dv/dz or dp/dz will be zero,
  // and only one of the terms of the addition above will be non-zero.
  //
  // We can determine which one by examining dv/dz (the field grad1).
  // If grad1 is not 0, then z is v,
  // and the log prob gradient will be d(log prob(v,p))/dv.
  // Otherwise, z is not v, dv/dz is 0,
  // and the log prob gradient will be d(log prob(v,p))/dp.

  const auto dist = static_cast<const distribution::Distribution*>(in_nodes[0]);
  if (grad1 != 0.0) {
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
  if (in_nodes.size() != 1 or
      in_nodes[0]->node_type != graph::NodeType::DISTRIBUTION) {
    throw std::invalid_argument(
        "~ operator requires a single distribution parent");
  }
  const distribution::Distribution* dist =
      static_cast<distribution::Distribution*>(in_nodes[0]);
  // the type of value of a SAMPLE node is obviously the sample type
  // of the distribution parent
  value = graph::NodeValue(dist->sample_type);
  unconstrained_value = graph::NodeValue(graph::ValueType(
      dist->sample_type.variable_type, graph::AtomicType::REAL, 1, 0));
}

IIdSample::IIdSample(const std::vector<graph::Node*>& in_nodes)
    : StochasticOperator(graph::OperatorType::IID_SAMPLE) {
  uint in_degree = in_nodes.size();
  if (in_degree != 2 and in_degree != 3) {
    throw std::invalid_argument(
        "iid sample operator requires 2 or 3 parent nodes");
  }
  if (in_nodes[0]->node_type != graph::NodeType::DISTRIBUTION) {
    throw std::invalid_argument(
        "for iid sample, the 1st parent must be a distribution node");
  }
  if (in_nodes[1]->node_type != graph::NodeType::CONSTANT or
      in_nodes[1]->value.type != graph::AtomicType::NATURAL) {
    throw std::invalid_argument(
        "for iid sample, the 2nd parent must be a constant natural-valued node");
  }
  if (in_degree == 3 and
      (in_nodes[2]->node_type != graph::NodeType::CONSTANT or
       in_nodes[2]->value.type != graph::AtomicType::NATURAL)) {
    throw std::invalid_argument(
        "for iid sample, the 3rd parent must be a constant natural-valued node");
  }
  if (in_degree == 2 and in_nodes[1]->value._natural < 2) {
    throw std::invalid_argument(
        "for iid sample with two parents, the 2nd parent must have value >= 2");
  }
  if (in_degree == 3 and
      (in_nodes[1]->value._natural * in_nodes[2]->value._natural) < 2) {
    throw std::invalid_argument(
        "for iid sample with three parents, the product of the 2nd and 3rd parents must be >= 2");
  }
  const auto dist = static_cast<distribution::Distribution*>(in_nodes[0]);

  // determine the value type
  graph::ValueType vtype;
  switch (dist->sample_type.variable_type) {
    case graph::VariableType::SCALAR:
      vtype = graph::ValueType(
          graph::VariableType::BROADCAST_MATRIX,
          dist->sample_type.atomic_type,
          in_nodes[1]->value._natural,
          in_degree == 2 ? 1 : in_nodes[2]->value._natural);
      break;
    case graph::VariableType::BROADCAST_MATRIX:
    case graph::VariableType::COL_SIMPLEX_MATRIX:
      // TODO(ddeng): add IID_SAMPLE_COL after first multivariate distrib added
      throw std::invalid_argument(
          "For matrix sample types, use IID_SAMPLE_COL. ");
    default:
      throw std::invalid_argument("Invalid sample type for for iid sample. ");
  }
  value = graph::NodeValue(vtype);
  unconstrained_value = graph::NodeValue(
      graph::ValueType(vtype.variable_type, graph::AtomicType::REAL, 1, 0));
  return;
}

} // namespace oper
} // namespace beanmachine
