/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include "beanmachine/graph/distribution/bernoulli.h"

namespace beanmachine {
namespace distribution {

Bernoulli::Bernoulli(
    graph::AtomicType sample_type,
    const std::vector<graph::Node*>& in_nodes)
    : Distribution(graph::DistributionType::BERNOULLI, sample_type) {
  if (sample_type != graph::AtomicType::BOOLEAN) {
    throw std::invalid_argument("Bernoulli produces boolean valued samples");
  }
  // a Bernoulli can only have one parent which must be a probability
  if (in_nodes.size() != 1) {
    throw std::invalid_argument(
        "Bernoulli distribution must have exactly one parent");
  }
  // check the parent type
  const auto& parent0 = in_nodes[0]->value;
  if (parent0.type != graph::AtomicType::PROBABILITY) {
    throw std::invalid_argument("Bernoulli parent must be a probability");
  }
}

bool Bernoulli::_bool_sampler(std::mt19937& gen) const {
  std::bernoulli_distribution distrib(in_nodes[0]->value._double);
  return (bool)distrib(gen);
}

double Bernoulli::log_prob(const graph::NodeValue& value) const {
  double prob = in_nodes[0]->value._double;

  if (value.type.variable_type == graph::VariableType::SCALAR) {
    return value._bool ? std::log(prob) : std::log(1 - prob);
  } else if (
      value.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    int size = static_cast<int>(value._bmatrix.size());
    int n_positive = static_cast<int>(value._bmatrix.count());
    return std::log(prob) * n_positive +
        std::log(1 - prob) * (size - n_positive);
  } else {
    throw std::runtime_error(
        "Bernoulli::log_prob applied to invalid variable type");
  }
}

void Bernoulli::log_prob_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& log_probs) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double prob = in_nodes[0]->value._double;
  double pos_val = std::log(prob);
  double neg_val = std::log(1 - prob);
  log_probs = Eigen::MatrixXd::Constant(
      value._bmatrix.rows(), value._bmatrix.cols(), neg_val);
  log_probs = value._bmatrix.select(pos_val, log_probs);
}

// The likelihood L(x|p) where x is the outcome and p is the parameter of
// the Bernoulli distribution is L(x|p) = p if x is 1, (1-p) if x is 0.
// We need the gradient of log(L(x|p)); how do we get a gradient on
// a discrete function?
//
// We find a continuous function that agrees with L(x|p) for x = 0 or 1
// and then differentiate that.  We choose L(x|p) = p^x * (1-p)^(1-x),
// so log(L(x|p)) = x * log(p) + (1-x) * log(1-p)
//
// grad1 wrt x = log(p) - log(1-p)
// grad2 wrt x = 0
// grad1 wrt p = (x/p) * p' - ((1-x)/(1-p)) * p'
// grad2 wrt p = -(x/p^2) * p'^2 + (x/p) * p'' - ((1-x)/(1-p)^2) * p'^2 -
//               ((1-x)/(1-p)) * p''
//
// First order chain rule: f(g(x))' = f'(g(x)) g'(x),
// - In backward propagation, f'(g(x)) is given by adjunct, the above equation
// computes g'(x). [g is the current function f is the final target]
// - In forward propagation, g'(x) is given by in_nodes[x]->grad1,
// the above equation computes f'(g) [f is the current function g is the input]
void Bernoulli::gradient_log_prob_value(
    const graph::NodeValue& /* value */,
    double& grad1,
    double& /* grad2 */) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double prob = in_nodes[0]->value._double;
  grad1 += std::log(prob) - std::log(1 - prob);
  // grad2 += 0
}

double Bernoulli::_grad1_log_prob_param(bool x, double p) {
  return x ? 1 / p : -1 / (1 - p);
}

void Bernoulli::gradient_log_prob_param(
    const graph::NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double val = value._bool ? 1.0 : 0.0;
  double prob = in_nodes[0]->value._double;
  double prob1 = in_nodes[0]->grad1;
  double prob2 = in_nodes[0]->grad2;
  grad1 += _grad1_log_prob_param(value._bool, prob) * prob1;
  grad2 += ((val / prob) - ((1 - val) / (1 - prob))) * prob2 +
      (-(val / (prob * prob)) - ((1 - val) / ((1 - prob) * (1 - prob)))) *
          prob1 * prob1;
}

void Bernoulli::backward_param(const graph::NodeValue& value, double adjunct)
    const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  if (in_nodes[0]->needs_gradient()) {
    bool x = value._bool;
    double prob = in_nodes[0]->value._double;
    in_nodes[0]->back_grad1._double += adjunct * _grad1_log_prob_param(x, prob);
  }
}

void Bernoulli::backward_param_iid(const graph::NodeValue& value) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  if (in_nodes[0]->needs_gradient()) {
    double prob = in_nodes[0]->value._double;
    int size = static_cast<int>(value._bmatrix.size());
    int n_positive = static_cast<int>(value._bmatrix.count());
    in_nodes[0]->back_grad1._double +=
        (1 / prob * n_positive - 1 / (1 - prob) * (size - n_positive));
  }
}

void Bernoulli::backward_param_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  if (in_nodes[0]->needs_gradient()) {
    double prob = in_nodes[0]->value._double;
    double sum_adjunct = adjunct.sum();
    double sum_pos_adjunct =
        (value._bmatrix.cast<double>().array() * adjunct.array()).sum();
    in_nodes[0]->back_grad1._double +=
        (1 / prob * sum_pos_adjunct -
         1 / (1 - prob) * (sum_adjunct - sum_pos_adjunct));
  }
}

} // namespace distribution
} // namespace beanmachine
