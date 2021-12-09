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

#include "beanmachine/graph/distribution/half_normal.h"

namespace beanmachine {
namespace distribution {

using namespace graph;

Half_Normal::Half_Normal(
    AtomicType sample_type,
    const std::vector<Node*>& in_nodes)
    : Distribution(DistributionType::HALF_NORMAL, sample_type) {
  // a Half_Normal distribution has one parent
  // sigma -> positive real
  if (in_nodes.size() != 1) {
    throw std::invalid_argument(
        "Half_Normal distribution must have exactly one parent");
  }
  if (in_nodes[0]->value.type != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "Half_Normal parent must be a positive real number");
  }
  // only real-valued samples are possible
  if (sample_type != AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "Half_Normal distribution produces positive real number samples");
  }
}

double Half_Normal::_double_sampler(std::mt19937& gen) const {
  std::normal_distribution<double> dist(0.0, in_nodes[0]->value._double);
  return std::abs(dist(gen));
}

// The calculations for half-normal distribution
// log_prob of a half-normal: - log(s) -0.5 log(pi/2) - 0.5 x^2 / s^2
// grad  w.r.t. value x: - x / s^2
// grad2 w.r.t. value x: - 1 / s^2
// grad  w.r.t. s : -1/s + x^2 / s^3
// grad2 w.r.t. s : 1/s^2 - 3 x^2 / s^4
// First order chain rule: f(g(x))' = f'(g(x)) g'(x),
// - In backward propagation, f'(g(x)) is given by adjunct, the above equation
// computes g'(x). [g is the current function f is the final target]
// - In forward propagation, g'(x) is given by in_nodes[x]->grad1,
// the above equation computes f'(g) [f is the current function g is the input]

/// TODO[Walid]: The following notes are mysterious to me.
// First order chain rule: f(g(x))' = f'(g(x)) g'(x),
// - In backward propagation, f'(g(x)) is given by adjunct, the above equation
// computes g'(x). [g is the current function f is the final target]
// - In forward propagation, g'(x) is given by in_nodes[x]->grad1,
// the above equation computes f'(g) [f is the current function g is the input]

double Half_Normal::log_prob(const NodeValue& value) const {
  double s = in_nodes[0]->value._double;
  double result, sum_xsq;
  int size;

  if (value.type.variable_type == graph::VariableType::SCALAR) {
    size = 1;
    sum_xsq = value._double * value._double;
  } else if (
      value.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    size = static_cast<int>(value._matrix.size());
    sum_xsq = value._matrix.squaredNorm();
  } else {
    throw std::runtime_error(
        "Half_Normal::log_prob applied to invalid variable type");
  }
  // This is computing the log probability of the Half Normal PDF
  // for the entire sample
  /// TODO[Walid]: Should we also add a check on mean?
  result = (-std::log(s) - 0.5 * std::log(M_PI / 2.0)) * size -
      0.5 * sum_xsq / (s * s);
  return result;
}

void Half_Normal::log_prob_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& log_probs) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double s = in_nodes[0]->value._double;
  /// TODO[Walid]: Need to figure out how to do constants and conditionals here
  log_probs = (-std::log(s) - 0.5 * std::log(M_PI / 2)) -
      0.5 * (value._matrix.array()).pow(2) / (s * s);
}

/// TODO[Walid]: This function can be inlined (it has only two uses)
/// TODO[Walid]: Will need to be conditioned by x either here or at call site
void Half_Normal::_grad1_log_prob_value(
    double& grad1,
    double val,
    double s_sq) {
  grad1 += -val / s_sq;
};

void Half_Normal::gradient_log_prob_value(
    const NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double s = in_nodes[0]->value._double;
  double s_sq = s * s;
  _grad1_log_prob_value(grad1, value._double, s_sq);
  grad2 += -1 / s_sq;
}

void Half_Normal::gradient_log_prob_param(
    const NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double x = value._double;
  double s = in_nodes[0]->value._double;
  double s_sq = s * s;
  double s_grad = in_nodes[0]->grad1;
  double s_grad2 = in_nodes[0]->grad2;
  if (s_grad != 0 or s_grad2 != 0) {
    double grad_s = -1 / s + x * x / (s * s * s);
    double grad2_s2 = 1 / s_sq - 3 * x * x / (s_sq * s_sq);
    grad1 += grad_s * s_grad;
    grad2 += grad2_s2 * s_grad * s_grad + grad_s * s_grad2;
  }
}

void Half_Normal::backward_value(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    double adjunct) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double s = in_nodes[0]->value._double;
  double s_sq = s * s;
  double increment = 0.0;
  _grad1_log_prob_value(increment, value._double, s_sq);
  back_grad._double += adjunct * increment;
}

void Half_Normal::backward_value_iid(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double s = in_nodes[0]->value._double;
  double s_sq = s * s;
  back_grad._matrix -= (value._matrix.array() / s_sq).matrix();
}

void Half_Normal::backward_value_iid(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double s = in_nodes[0]->value._double;
  double s_sq = s * s;
  back_grad._matrix -=
      (adjunct.array() * value._matrix.array() / s_sq).matrix();
}

void Half_Normal::backward_param(const graph::NodeValue& value, double adjunct)
    const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double s = in_nodes[0]->value._double;
  double s_sq = s * s;
  double jacob_0 = value._double / s_sq;

  /// Delete the following
  /// if (in_nodes[0]->needs_gradient()) {
  ///  in_nodes[0]->back_grad1._double += adjunct * jacob_0;
  /// }
  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1._double +=
        adjunct * (-1 / s + jacob_0 * jacob_0 * s);
  }
}

void Half_Normal::backward_param_iid(const graph::NodeValue& value) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double s = in_nodes[0]->value._double;
  double s_sq = s * s;

  int size = static_cast<int>(value._matrix.size());
  /// The following should be deleted
  ///  if (in_nodes[0]->needs_gradient()) {
  ///    in_nodes[0]->back_grad1._double += sum_x / s_sq - size * m / s_sq;
  ///  }
  if (in_nodes[0]->needs_gradient()) {
    double sum_xsq = value._matrix.squaredNorm();
    in_nodes[0]->back_grad1._double += (-size / s + sum_xsq / (s * s_sq));
  }
}

void Half_Normal::backward_param_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double s = in_nodes[0]->value._double;
  double s_sq = s * s;

  double sum_adjunct = adjunct.sum();
  /// The following should be deleted
  /// if (in_nodes[0]->needs_gradient()) {
  ///  in_nodes[0]->back_grad1._double += sum_x / s_sq - sum_adjunct * m / s_sq;
  /// }
  if (in_nodes[0]->needs_gradient()) {
    double sum_xsq = (value._matrix.array().pow(2) * adjunct.array()).sum();
    in_nodes[0]->back_grad1._double +=
        (-sum_adjunct / s + sum_xsq / (s * s_sq));
  }
}

} // namespace distribution
} // namespace beanmachine
