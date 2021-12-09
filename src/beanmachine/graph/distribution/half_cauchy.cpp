/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <random>
#include <string>

#include "beanmachine/graph/distribution/half_cauchy.h"

namespace beanmachine {
namespace distribution {

using namespace graph;

HalfCauchy::HalfCauchy(
    AtomicType sample_type,
    const std::vector<Node*>& in_nodes)
    : Distribution(DistributionType::HALF_CAUCHY, sample_type) {
  // a HalfCauchy distribution has one parent a scale which is positive real
  if (in_nodes.size() != 1) {
    throw std::invalid_argument(
        "HalfCauchy distribution must have exactly one parent");
  }
  if (in_nodes[0]->value.type != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "HalfCauchy parent must be positive real number");
  }
  // only positive real-valued samples are possible
  if (sample_type != AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "HalfCauchy distribution produces positive real number samples");
  }
}

double HalfCauchy::_double_sampler(std::mt19937& gen) const {
  // the cdf of a standard HalfCauchy is  F(x) = (2/pi) arctan(x)
  // therefore we will sample w ~ uniformly [0, pi/2] and compute tan(w)
  // finally we will multiply the scale to get the required value
  std::uniform_real_distribution<double> dist(0.0, M_PI_2);
  return in_nodes[0]->value._double * std::tan(dist(gen));
}

// log_prob of a HalfCauchy is f(x; s) =  -log(pi/2) -log(s) -log(1 + (x/s)^2)
//   = -log(pi/2) + log(s) - log(s^2 + x^2)
// df/dx = -2x/(s^2 + x^2)
// d2f/dx2 = -2/(s^2 + x^2) + 4x^2/(s^2 + x^2)^2
// df/ds = 1/s -2s/(s^2 + x^2)
// d2f/ds2 = - 1/s^2 - 2/(s^2 + x^2) + 4s^2/(s^2 + x^2)^2
// First order chain rule: f(g(x))' = f'(g(x)) g'(x),
// - In backward propagation, f'(g(x)) is given by adjunct, the above equation
// computes g'(x). [g is the current function f is the final target]
// - In forward propagation, g'(x) is given by in_nodes[x]->grad1,
// the above equation computes f'(g) [f is the current function g is the input]
double HalfCauchy::log_prob(const NodeValue& value) const {
  double s = in_nodes[0]->value._double;
  double result;
  int size;

  if (value.type.variable_type == graph::VariableType::SCALAR) {
    size = 1;
    result = std::log1p(std::pow(value._double / s, 2));
  } else if (
      value.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    size = static_cast<int>(value._matrix.size());
    result = (value._matrix.array() / s).pow(2).log1p().sum();
  } else {
    throw std::runtime_error(
        "HalfCauchy::log_prob applied to invalid variable type");
  }
  return (-std::log(M_PI_2) - std::log(s)) * size - result;
}

void HalfCauchy::log_prob_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& log_probs) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double s = in_nodes[0]->value._double;
  log_probs = -std::log(M_PI_2) - std::log(s) -
      (value._matrix.array() / s).pow(2).log1p();
}

void HalfCauchy::_grad1_log_prob_value(
    double& grad1,
    double val,
    double s2_p_x2) {
  grad1 += -2 * val / s2_p_x2;
}

void HalfCauchy::gradient_log_prob_value(
    const NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double x = value._double;
  double s = in_nodes[0]->value._double;
  double s2_p_x2 = s * s + x * x;
  _grad1_log_prob_value(grad1, x, s2_p_x2);
  grad2 += -2 / s2_p_x2 + 4 * x * x / (s2_p_x2 * s2_p_x2);
}

double HalfCauchy::_grad1_log_prob_param(double s, double s2_p_x2) {
  return 1 / s - 2 * s / s2_p_x2;
}

void HalfCauchy::gradient_log_prob_param(
    const NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  // gradients of s should be non-zero before computing gradients w.r.t. s
  double s_grad = in_nodes[0]->grad1;
  double s_grad2 = in_nodes[0]->grad2;
  if (s_grad != 0 or s_grad2 != 0) {
    double x = value._double;
    double s = in_nodes[0]->value._double;
    double s2_p_x2 = s * s + x * x;
    double grad_s = _grad1_log_prob_param(s, s2_p_x2);
    double grad2_s2 =
        -1 / (s * s) - 2 / s2_p_x2 + 4 * s * s / (s2_p_x2 * s2_p_x2);
    grad1 += grad_s * s_grad;
    grad2 += grad2_s2 * s_grad * s_grad + grad_s * s_grad2;
  }
}

void HalfCauchy::backward_value(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    double adjunct) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double x = value._double;
  double s = in_nodes[0]->value._double;
  double s2_p_x2 = s * s + x * x;
  double increment = 0.0;
  _grad1_log_prob_value(increment, x, s2_p_x2);
  back_grad._double += adjunct * increment;
}

void HalfCauchy::backward_value_iid(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double s = in_nodes[0]->value._double;
  Eigen::MatrixXd s2_p_x2 =
      s * s + value._matrix.array() * value._matrix.array();
  back_grad._matrix -= (2 * value._matrix.array() / s2_p_x2.array()).matrix();
}

void HalfCauchy::backward_value_iid(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double s = in_nodes[0]->value._double;
  Eigen::MatrixXd s2_p_x2 =
      s * s + value._matrix.array() * value._matrix.array();
  back_grad._matrix -=
      (2 * adjunct.array() * value._matrix.array() / s2_p_x2.array()).matrix();
}

void HalfCauchy::backward_param(const graph::NodeValue& value, double adjunct)
    const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  if (in_nodes[0]->needs_gradient()) {
    double x = value._double;
    double s = in_nodes[0]->value._double;
    double s2_p_x2 = s * s + x * x;
    in_nodes[0]->back_grad1._double +=
        adjunct * _grad1_log_prob_param(s, s2_p_x2);
  }
}

void HalfCauchy::backward_param_iid(const graph::NodeValue& value) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  if (in_nodes[0]->needs_gradient()) {
    int size = static_cast<int>(value._matrix.size());
    double s = in_nodes[0]->value._double;
    Eigen::MatrixXd s2_p_x2 =
        s * s + value._matrix.array() * value._matrix.array();
    in_nodes[0]->back_grad1._double +=
        size / s - 2 * (s / s2_p_x2.array()).sum();
  }
}

void HalfCauchy::backward_param_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  if (in_nodes[0]->needs_gradient()) {
    double s = in_nodes[0]->value._double;
    double sum_adjunct = adjunct.sum();
    Eigen::MatrixXd s2_p_x2 =
        s * s + value._matrix.array() * value._matrix.array();
    in_nodes[0]->back_grad1._double +=
        sum_adjunct / s - 2 * s * (adjunct.array() / s2_p_x2.array()).sum();
  }
}

} // namespace distribution
} // namespace beanmachine
