/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include "beanmachine/graph/distribution/geometric.h"

namespace beanmachine {
namespace distribution {

Geometric::Geometric(
    graph::AtomicType sample_type,
    const std::vector<graph::Node*>& in_nodes)
    : Distribution(graph::DistributionType::GEOMETRIC, sample_type) {
  if (sample_type != graph::AtomicType::NATURAL) {
    throw std::invalid_argument("Geometric produces natural valued samples");
  }
  // a geometric can only have one parent which is the success probability
  if (in_nodes.size() != 1) {
    throw std::invalid_argument(
        "Geometric distribution must have exactly one parent");
  }
  // check the parent type
  const auto& parent0 = in_nodes[0]->value;
  if (parent0.type != graph::AtomicType::PROBABILITY) {
    throw std::invalid_argument("Geometric parent must be a probability");
  }
}

graph::natural_t Geometric::_natural_sampler(std::mt19937& gen) const {
  std::geometric_distribution<graph::natural_t> distrib(
      in_nodes[0]->value._double);
  return distrib(gen);
}

// log_prob(k | p) = k * log(1-p) + log(p)
// grad1 w.r.t k   = log(1-p)
// grad2 w.r.t k   = 0
// grad1 w.r.t p   = p' (1/p - k/(1-p))
// grad2 w.r.t p   = p'' ( 1/p - k/(1-p)) + p'^2 ( -1/p^2 - k/(1-p)^2)
double Geometric::log_prob(const graph::NodeValue& value) const {
  double p = in_nodes[0]->value._double;
  double ret_val = 0;
  if (value.type.variable_type == graph::VariableType::SCALAR) {
    graph::natural_t k = value._natural;
    if (k < 0) {
      return -std::numeric_limits<double>::infinity();
    }
    ret_val += k * log1p(-p) + log(p);
  } else if (
      value.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    if ((value._nmatrix.array() < 0).any()) {
      return -std::numeric_limits<double>::infinity();
    }
    int size = static_cast<int>(value._nmatrix.size());
    Eigen::MatrixXd k_double = value._nmatrix.cast<double>();

    double k_log_1_minus_p_sum = k_double.sum() * log1p(-p);

    ret_val += k_log_1_minus_p_sum + size * log(p);
  }
  return ret_val;
}

void Geometric::log_prob_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& log_probs) const {
  double p = in_nodes[0]->value._double;
  Eigen::MatrixXd k_double = value._nmatrix.cast<double>();
  log_probs = k_double.array() * log1p(-p) + log(p);
}

void Geometric::gradient_log_prob_param(
    const graph::NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  graph::natural_t k = value._natural;
  double p = in_nodes[0]->value._double;
  double grad_p = in_nodes[0]->grad1;
  double grad2_p2 = in_nodes[0]->grad2;
  grad1 += grad_p * (1 / p - k / (1 - p));
  grad2 += grad2_p2 * (1 / p - k / (1 - p)) +
      grad_p * grad_p * (-1 / (p * p) - k / ((1 - p) * (1 - p)));
}

void Geometric::backward_param(const graph::NodeValue& value, double adjunct)
    const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  if (in_nodes[0]->needs_gradient()) {
    double p = in_nodes[0]->value._double;
    graph::natural_t k = value._natural;
    in_nodes[0]->back_grad1 += adjunct * (1 / p - k / (1 - p));
  }
}
void Geometric::backward_param_iid(const graph::NodeValue& value) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  if (in_nodes[0]->needs_gradient()) {
    double p = in_nodes[0]->value._double;
    Eigen::MatrixXd k_double = value._nmatrix.cast<double>();
    in_nodes[0]->back_grad1 += (1 / p - k_double.array() / (1 - p)).sum();
  }
}

void Geometric::backward_param_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  if (in_nodes[0]->needs_gradient()) {
    double p = in_nodes[0]->value._double;
    Eigen::MatrixXd k_double = value._nmatrix.cast<double>();
    in_nodes[0]->back_grad1 +=
        (adjunct.array() * (1 / p - k_double.array() / (1 - p))).sum();
  }
}

} // namespace distribution
} // namespace beanmachine
