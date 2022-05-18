/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <unsupported/Eigen/SpecialFunctions>

#include "beanmachine/graph/distribution/poisson.h"

namespace beanmachine {
namespace distribution {

Poisson::Poisson(
    graph::AtomicType sample_type,
    const std::vector<graph::Node*>& in_nodes)
    : Distribution(graph::DistributionType::POISSON, sample_type) {
  if (sample_type != graph::AtomicType::NATURAL) {
    throw std::invalid_argument("Poisson produces natural valued samples");
  }
  // a poisson can only have one parent which is the mean (lambda)
  if (in_nodes.size() != 1) {
    throw std::invalid_argument(
        "Poisson distribution must have exactly one parent");
  }
  // check the parent type
  const auto& parent0 = in_nodes[0]->value;
  if (parent0.type != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument("Poisson parent must be a positive real");
  }
}

graph::natural_t Poisson::_natural_sampler(std::mt19937& gen) const {
  std::poisson_distribution<graph::natural_t> distrib(
      in_nodes[0]->value._double);
  return distrib(gen);
}

// log_prob(k | lambda) = k * log(lambda) - lambda - log(k!)
// equivalently         = k * log(lambda) - lambda - lgamma(k+1)
// grad1 w.r.t k        = log(lambda) - digamma(k+1)
// grad2 w.r.t k        = -polygamma(1, k+1)
// grad1 w.r.t lambda   = (k/lambda - 1) * lambda'
// grad2 w.r.t lambda   = (k/lambda - 1) * lambda'' - (k/lambda^2) * lambda'^2
double Poisson::log_prob(const graph::NodeValue& value) const {
  double lambda = in_nodes[0]->value._double;
  double ret_val = 0;
  if (value.type.variable_type == graph::VariableType::SCALAR) {
    graph::natural_t k = value._natural;
    if (k < 0) {
      return -std::numeric_limits<double>::infinity();
    }

    ret_val += k * log(lambda) - lambda - std::lgamma(k + 1);
  } else if (
      value.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    if ((value._nmatrix.array() < 0).any()) {
      return -std::numeric_limits<double>::infinity();
    }
    int size = static_cast<int>(value._nmatrix.size());
    Eigen::MatrixXd k_double = value._nmatrix.cast<double>();

    double k_factorial_sum = (k_double.array() + 1).lgamma().sum();
    double k_log_lambda_sum = k_double.sum() * log(lambda);
    ret_val += k_log_lambda_sum - size * lambda - k_factorial_sum;
  }
  return ret_val;
}

void Poisson::log_prob_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& log_probs) const {
  double lambda = in_nodes[0]->value._double;
  Eigen::MatrixXd k_double = value._nmatrix.cast<double>();
  log_probs =
      k_double.array() * log(lambda) - lambda - (k_double.array() + 1).lgamma();
}

void Poisson::gradient_log_prob_param(
    const graph::NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  graph::natural_t k = value._natural;
  double lambda = in_nodes[0]->value._double;
  double grad_lambda = in_nodes[0]->grad1;
  double grad2_lambda2 = in_nodes[0]->grad2;
  grad1 += (k / lambda - 1) * grad_lambda;
  grad2 += (k / lambda - 1) * grad2_lambda2 -
      k * grad_lambda * grad_lambda / (lambda * lambda);
}

void Poisson::backward_param(const graph::NodeValue& value, double adjunct)
    const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  if (in_nodes[0]->needs_gradient()) {
    double lambda = in_nodes[0]->value._double;
    graph::natural_t k = value._natural;
    in_nodes[0]->back_grad1 += adjunct * (k / lambda - 1);
  }
}

void Poisson::backward_param_iid(const graph::NodeValue& value) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  if (in_nodes[0]->needs_gradient()) {
    double lambda = in_nodes[0]->value._double;
    Eigen::MatrixXd k_double = value._nmatrix.cast<double>();
    in_nodes[0]->back_grad1 += (k_double.array() / lambda - 1).sum();
  }
}

void Poisson::backward_param_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  if (in_nodes[0]->needs_gradient()) {
    double lambda = in_nodes[0]->value._double;
    Eigen::MatrixXd k_double = value._nmatrix.cast<double>();
    in_nodes[0]->back_grad1 +=
        (adjunct.array() * (k_double.array() / lambda - 1)).sum();
  }
}

} // namespace distribution
} // namespace beanmachine
