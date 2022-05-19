/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define _USE_MATH_DEFINES
#include <Eigen/Core>
#include <cmath>
#include <random>

#include "beanmachine/graph/distribution/log_normal.h"

namespace beanmachine {
namespace distribution {

using namespace graph;

LogNormal::LogNormal(AtomicType sample_type, const std::vector<Node*>& in_nodes)
    : Distribution(DistributionType::LOG_NORMAL, sample_type) {
  // a Log Normal distribution has two parents
  // mean of logarithm distribution -> real,
  // standard deviation of logarithm distribution -> positive real
  if (in_nodes.size() != 2) {
    throw std::invalid_argument(
        "LogNormal distribution must have exactly two parents");
  }
  if (in_nodes[0]->value.type != graph::AtomicType::REAL or
      in_nodes[1]->value.type != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "LogNormal parents must be a real number and a positive real number");
  }
  if (sample_type != AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "LogNormal distribution produces positive real number samples");
  }
}

double LogNormal::_double_sampler(std::mt19937& gen) const {
  std::lognormal_distribution<double> dist(
      in_nodes[0]->value._double, in_nodes[1]->value._double);
  return dist(gen);
}

// log_prob of a log normal:
//    - log(s) - 0.5 log(2*pi) - 0.5 (log(x) - m)^2 / s^2 - log(x)
// grad  w.r.t. value x: (m - log(x) - s^2) / (x * s^2)
// grad2 w.r.t. value x: (s^2 + log(x) - m - 1) / (s^2 * x^2)
// grad  w.r.t. s : -1/s + (log(x)-m)^2 / s^3
// grad2 w.r.t. s : 1/s^2 - 3 (log(x)-m)^2 / s^4
// grad  w.r.t. m : (log(x) - m) / s^2
// grad2 w.r.t. m : -1 / s^2
// First order chain rule: f(g(x))' = f'(g(x)) g'(x),
// - In backward propagation, f'(g(x)) is given by adjunct, the above equation
// computes g'(x). [g is the current function f is the final target]
// - In forward propagation, g'(x) is given by in_nodes[x]->grad1,
// the above equation computes f'(g) [f is the current function g is the input]
double LogNormal::log_prob(const NodeValue& value) const {
  double m = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  double result, sum_logx, sum_logx_sq;
  int size;

  if (value.type.variable_type == graph::VariableType::SCALAR) {
    size = 1;
    sum_logx = std::log(value._double);
    sum_logx_sq = sum_logx * sum_logx;
  } else if (
      value.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    size = static_cast<int>(value._matrix.size());
    sum_logx = value._matrix.array().log().matrix().sum();
    sum_logx_sq = value._matrix.array().log().matrix().squaredNorm();
  } else {
    throw std::runtime_error(
        "LogNormal::log_prob applied to invalid variable type");
  }
  result = (-std::log(s) - 0.5 * std::log(2 * M_PI)) * size -
      0.5 * (sum_logx_sq - 2 * m * sum_logx + m * m * size) / (s * s) -
      sum_logx;
  return result;
}

void LogNormal::log_prob_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& log_probs) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double m = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> logs =
      value._matrix.array().log();
  log_probs = (-std::log(s) - 0.5 * std::log(2 * M_PI)) -
      0.5 * (logs - m).pow(2) / (s * s) - logs;
}

void LogNormal::_grad1_log_prob_value(
    double& grad1,
    double val,
    double m,
    double s_sq) {
  grad1 += (m - std::log(val) - s_sq) / (val * s_sq);
};

void LogNormal::_grad2_log_prob_value(
    double& grad2,
    double val,
    double m,
    double s_sq) {
  grad2 += (s_sq + std::log(val) - m - 1) / (val * val * s_sq);
};

void LogNormal::gradient_log_prob_value(
    const NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double m = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  double s_sq = s * s;
  _grad1_log_prob_value(grad1, value._double, m, s_sq);
  _grad2_log_prob_value(grad2, value._double, m, s_sq);
}

void LogNormal::gradient_log_prob_param(
    const NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double log_x = std::log(value._double);
  double m = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  double s_sq = s * s;
  // gradients of m should be non-zero before computing gradients w.r.t. m
  double m_grad = in_nodes[0]->grad1;
  double m_grad2 = in_nodes[0]->grad2;
  if (m_grad != 0 or m_grad2 != 0) {
    double grad_m = (log_x - m) / s_sq;
    double grad2_m2 = -1 / s_sq;
    grad1 += grad_m * m_grad;
    grad2 += grad2_m2 * m_grad * m_grad + grad_m * m_grad2;
  }
  double s_grad = in_nodes[1]->grad1;
  double s_grad2 = in_nodes[1]->grad2;
  if (s_grad != 0 or s_grad2 != 0) {
    double grad_s = -1 / s + (log_x - m) * (log_x - m) / (s * s * s);
    double grad2_s2 = 1 / s_sq - 3 * (log_x - m) * (log_x - m) / (s_sq * s_sq);
    grad1 += grad_s * s_grad;
    grad2 += grad2_s2 * s_grad * s_grad + grad_s * s_grad2;
  }
}

void LogNormal::backward_value(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    double adjunct) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double m = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  double s_sq = s * s;
  double increment = 0.0;
  _grad1_log_prob_value(increment, value._double, m, s_sq);
  back_grad += adjunct * increment;
}

void LogNormal::backward_value_iid(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double m = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  double s_sq = s * s;
  back_grad +=
      (m - value._matrix.array().log() - s_sq) / (value._matrix.array() * s_sq);
}

void LogNormal::backward_value_iid(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double m = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  double s_sq = s * s;
  back_grad += (adjunct.array()) * (m - value._matrix.array().log() - s_sq) /
      (value._matrix.array() * s_sq);
}

void LogNormal::backward_param(const graph::NodeValue& value, double adjunct)
    const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double m = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  double log_x = std::log(value._double);
  double s_sq = s * s;
  double jacob_0 = (log_x - m) / s_sq;

  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1 += adjunct * jacob_0;
  }
  if (in_nodes[1]->needs_gradient()) {
    in_nodes[1]->back_grad1 += adjunct * (-1 / s + jacob_0 * jacob_0 * s);
  }
}

void LogNormal::backward_param_iid(const graph::NodeValue& value) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double m = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  double s_sq = s * s;

  int size = static_cast<int>(value._matrix.size());
  double sum_logx = value._matrix.array().log().sum();
  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1 += sum_logx / s_sq - size * m / s_sq;
  }
  if (in_nodes[1]->needs_gradient()) {
    double sum_logx_sq = value._matrix.array().log().matrix().squaredNorm();
    in_nodes[1]->back_grad1 +=
        (-size / s +
         (sum_logx_sq - 2 * m * sum_logx + m * m * size) / (s * s_sq));
  }
}

void LogNormal::backward_param_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double m = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  double s_sq = s * s;

  double sum_logx = (value._matrix.array().log() * adjunct.array()).sum();
  double sum_adjunct = adjunct.sum();
  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1 += sum_logx / s_sq - sum_adjunct * m / s_sq;
  }
  if (in_nodes[1]->needs_gradient()) {
    double sum_logx_sq =
        (value._matrix.array().log().pow(2) * adjunct.array()).sum();
    in_nodes[1]->back_grad1 +=
        (-sum_adjunct / s +
         (sum_logx_sq - 2 * m * sum_logx + m * m * sum_adjunct) / (s * s_sq));
  }
}

} // namespace distribution
} // namespace beanmachine
