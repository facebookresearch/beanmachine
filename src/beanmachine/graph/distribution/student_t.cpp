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

#include "beanmachine/graph/distribution/student_t.h"
#include "beanmachine/graph/util.h"

// the common steps for all gradient calculation
#define T_PREPARE_GRAD()                 \
  double x = value._double;              \
  double n = in_nodes[0]->value._double; \
  double l = in_nodes[1]->value._double; \
  double s = in_nodes[2]->value._double; \
  double n_s_sq_p_x_m_l_sq = n * s * s + (x - l) * (x - l);

// the matrix form of n s^2 + (x - l)^2
#define NS2PXML2 ((value._matrix.array() - l).pow(2) + n * s * s)

namespace beanmachine {
namespace distribution {

using namespace graph;

StudentT::StudentT(AtomicType sample_type, const std::vector<Node*>& in_nodes)
    : Distribution(DistributionType::STUDENT_T, sample_type) {
  // a StudentT distribution has three parents
  // n (degrees of freedom) > 0 ; l (location) -> real; scale -> positive real
  if (in_nodes.size() != 3) {
    throw std::invalid_argument(
        "StudentT distribution must have exactly three parents");
  }
  if (in_nodes[0]->value.type != graph::AtomicType::POS_REAL or
      in_nodes[1]->value.type != graph::AtomicType::REAL or
      in_nodes[2]->value.type != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "StudentT parents must have parents (positive, real, positive)");
  }
  // only real-valued samples are possible
  if (sample_type != AtomicType::REAL) {
    throw std::invalid_argument(
        "StudentT distribution produces real number samples");
  }
}

double StudentT::_double_sampler(std::mt19937& gen) const {
  double n = in_nodes[0]->value._double;
  double l = in_nodes[1]->value._double;
  double s = in_nodes[2]->value._double;
  std::student_t_distribution<double> dist(n);
  return l + dist(gen) * s;
}

// log_prob of a student-t with parameters n, l, s
// (degrees of freedom, location, and scale respectively):
// f(x, n, l, s) =
//     log(G((n+1)/2)) - log(G(n/2)) - 0.5 log(n) - 0.5 log(pi) - log(s)
//     -(n+1)/2 [ log(n s^2 + (x-l)^2) - log(n) - 2 log(s) ]
//
// df / dx = = -(n+1) (x-l) / (n s^2 + (x-l)^2)
// d2f/dx2 = -(n+1) [ 1/(n s^2 + (x-l)^2)   -  2(x-l)^2 / (n s^2 + (x-l)^2)^2 ]
//
// df / dn = 0.5 digamma((n+1)/2) - 0.5 digamma(n/2) - 0.5/n
//           -0.5 [ log(n s^2 + (x-l)^2) - log(n) - 2 log(s) ]
//           -0.5 (n+1) [s^2/(n s^2 + (x-l)^2) - 1/n ]
// d2f/dn2 = 0.25 polygamma(1, (n+1)/2) - 0.25 * polygamma(1, n/2) + 0.5/n^2
//           -0.5 [ s^2 / (n s^2 + (x-l)^2) - 1/n ]
//           -0.5 [ s^2 / (n s^2 + (x-l)^2) - 1/n ]
//           -0.5 (n+1) [ -s^4 / (n s^2 + (x-l)^2)^2 + 1/n^2 ]
// df / dl   = (n+1) (x-l) / (n s^2 + (x-l)^2)
// d2f / dl2 = - (n+1) / (n s^2 + (x-l)^2) + 2 (n+1) (x-l)^2 / (n s^2 +
// (x-l)^2)^2 df / ds  = -1/s -(n+1) ( n s / (n s^2 + (x-l)^2)  - 1/s ) d2f ds2
// = 1/s^2 -(n+1)( n / (n s^2 + (x-l)^2) - 2 n^2 s^2 / (n s^2 + (x-l)^2)^2  +
// 1/s^2)
// First order chain rule: f(g(x))' = f'(g(x)) g'(x),
// - In backward propagation, f'(g(x)) is given by adjunct, the above equation
// computes g'(x). [g is the current function f is the final target]
// - In forward propagation, g'(x) is given by in_nodes[x]->grad1,
// the above equation computes f'(g) [f is the current function g is the input]

double StudentT::log_prob(const NodeValue& value) const {
  double n = in_nodes[0]->value._double;
  double l = in_nodes[1]->value._double;
  double s = in_nodes[2]->value._double;
  double result = std::lgamma((n + 1) / 2) - std::lgamma(n / 2) -
      0.5 * std::log(n) - 0.5 * std::log(M_PI) - std::log(s) +
      ((n + 1) / 2) * (std::log(n) + 2 * std::log(s));

  if (value.type.variable_type == graph::VariableType::SCALAR) {
    double x = value._double;
    result -= ((n + 1) / 2) * std::log(n * s * s + (x - l) * (x - l));
  } else if (
      value.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    int size = static_cast<int>(value._matrix.size());
    result = result * size - ((n + 1) / 2) * NS2PXML2.log().sum();
  } else {
    throw std::runtime_error(
        "StudentT::log_prob applied to invalid variable type");
  }
  return result;
}

void StudentT::log_prob_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& log_probs) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double n = in_nodes[0]->value._double;
  double l = in_nodes[1]->value._double;
  double s = in_nodes[2]->value._double;
  double result = std::lgamma((n + 1) / 2) - std::lgamma(n / 2) -
      0.5 * std::log(n) - 0.5 * std::log(M_PI) - std::log(s) +
      ((n + 1) / 2) * (std::log(n) + 2 * std::log(s));

  log_probs = result - ((n + 1) / 2) * NS2PXML2.log();
}

void StudentT::_grad1_log_prob_value(
    double& grad1,
    double x,
    double n,
    double l,
    double n_s_sq_p_x_m_l_sq) {
  grad1 += -(n + 1) * (x - l) / n_s_sq_p_x_m_l_sq;
}

void StudentT::gradient_log_prob_value(
    const NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  T_PREPARE_GRAD()
  _grad1_log_prob_value(grad1, x, n, l, n_s_sq_p_x_m_l_sq);
  grad2 += -(n + 1) *
      (1 / n_s_sq_p_x_m_l_sq -
       2 * (x - l) * (x - l) / (n_s_sq_p_x_m_l_sq * n_s_sq_p_x_m_l_sq));
}

double
StudentT::_grad1_log_prob_n(double n, double s, double n_s_sq_p_x_m_l_sq) {
  return 0.5 * util::polygamma(0, (n + 1) / 2) -
      0.5 * util::polygamma(0, n / 2) - 0.5 / n -
      0.5 * (std::log(n_s_sq_p_x_m_l_sq) - std::log(n) - 2 * std::log(s)) -
      0.5 * (n + 1) * (s * s / n_s_sq_p_x_m_l_sq - 1 / n);
}

double StudentT::_grad1_log_prob_l(
    double x,
    double n,
    double l,
    double n_s_sq_p_x_m_l_sq) {
  return (n + 1) * (x - l) / n_s_sq_p_x_m_l_sq;
}

double
StudentT::_grad1_log_prob_s(double n, double s, double n_s_sq_p_x_m_l_sq) {
  return -1 / s - (n + 1) * (n * s / n_s_sq_p_x_m_l_sq - 1 / s);
}

void StudentT::gradient_log_prob_param(
    const NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  T_PREPARE_GRAD()
  // We will compute the gradients w.r.t. each of the parameters only if
  // the gradients of the parameters w.r.t. the source index is non-zero
  double n_grad = in_nodes[0]->grad1;
  double n_grad2 = in_nodes[0]->grad2;
  if (n_grad != 0 or n_grad2 != 0) {
    double grad_n = _grad1_log_prob_n(n, s, n_s_sq_p_x_m_l_sq);
    double grad2_n2 = 0.25 * util::polygamma(1, (n + 1) / 2) -
        0.25 * util::polygamma(1, n / 2) + 0.5 / (n * n) -
        (s * s / n_s_sq_p_x_m_l_sq - 1 / n) -
        0.5 * (n + 1) *
            (-std::pow(s, 4) / (n_s_sq_p_x_m_l_sq * n_s_sq_p_x_m_l_sq) +
             1 / (n * n));
    grad1 += grad_n * n_grad;
    grad2 += grad2_n2 * n_grad * n_grad + grad_n * n_grad2;
  }
  double l_grad = in_nodes[1]->grad1;
  double l_grad2 = in_nodes[1]->grad2;
  if (l_grad != 0 or l_grad2 != 0) {
    double grad_l = _grad1_log_prob_l(x, n, l, n_s_sq_p_x_m_l_sq);
    double grad2_l2 = -(n + 1) / n_s_sq_p_x_m_l_sq +
        2 * (n + 1) * (x - l) * (x - l) /
            (n_s_sq_p_x_m_l_sq * n_s_sq_p_x_m_l_sq);
    grad1 += grad_l * l_grad;
    grad2 += grad2_l2 * l_grad * l_grad + grad_l * l_grad2;
  }
  double s_grad = in_nodes[2]->grad1;
  double s_grad2 = in_nodes[2]->grad2;
  if (s_grad != 0 or s_grad2 != 0) {
    double grad_s = _grad1_log_prob_s(n, s, n_s_sq_p_x_m_l_sq);
    double grad2_s2 = 1 / (s * s) -
        (n + 1) *
            (n / n_s_sq_p_x_m_l_sq -
             2 * n * n * s * s / (n_s_sq_p_x_m_l_sq * n_s_sq_p_x_m_l_sq) +
             1 / (s * s));
    grad1 += grad_s * s_grad;
    grad2 += grad2_s2 * s_grad * s_grad + grad_s * s_grad2;
  }
}

void StudentT::backward_value(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    double adjunct) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  T_PREPARE_GRAD()
  double increment = 0.0;
  _grad1_log_prob_value(increment, x, n, l, n_s_sq_p_x_m_l_sq);
  back_grad._double += adjunct * increment;
}

void StudentT::backward_value_iid(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double n = in_nodes[0]->value._double;
  double l = in_nodes[1]->value._double;
  double s = in_nodes[2]->value._double;
  back_grad._matrix -=
      ((n + 1) * (value._matrix.array() - l) / NS2PXML2).matrix();
}

void StudentT::backward_value_iid(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double n = in_nodes[0]->value._double;
  double l = in_nodes[1]->value._double;
  double s = in_nodes[2]->value._double;
  back_grad._matrix -=
      (adjunct.array() * (n + 1) * (value._matrix.array() - l) / NS2PXML2)
          .matrix();
}

void StudentT::backward_param(const graph::NodeValue& value, double adjunct)
    const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  T_PREPARE_GRAD()
  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1._double +=
        adjunct * _grad1_log_prob_n(n, s, n_s_sq_p_x_m_l_sq);
  }
  if (in_nodes[1]->needs_gradient()) {
    in_nodes[1]->back_grad1._double +=
        adjunct * _grad1_log_prob_l(x, n, l, n_s_sq_p_x_m_l_sq);
  }
  if (in_nodes[2]->needs_gradient()) {
    in_nodes[2]->back_grad1._double +=
        adjunct * _grad1_log_prob_s(n, s, n_s_sq_p_x_m_l_sq);
  }
}

void StudentT::backward_param_iid(const graph::NodeValue& value) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double n = in_nodes[0]->value._double;
  double l = in_nodes[1]->value._double;
  double s = in_nodes[2]->value._double;
  Eigen::MatrixXd NSsqPXMLsq = NS2PXML2;
  int size = static_cast<int>(value._matrix.size());
  if (in_nodes[0]->needs_gradient()) {
    double jacob = size *
        (0.5 * util::polygamma(0, (n + 1) / 2) -
         0.5 * util::polygamma(0, n / 2) - 0.5 / n + 0.5 * std::log(n) +
         std::log(s) + 0.5 * (n + 1) / n);
    in_nodes[0]->back_grad1._double += jacob -
        (0.5 * NSsqPXMLsq.array().log() +
         0.5 * (n + 1) * s * s / NSsqPXMLsq.array())
            .sum();
  }
  if (in_nodes[1]->needs_gradient()) {
    in_nodes[1]->back_grad1._double +=
        (n + 1) * ((value._matrix.array() - l) / NSsqPXMLsq.array()).sum();
  }
  if (in_nodes[2]->needs_gradient()) {
    in_nodes[2]->back_grad1._double +=
        size * n / s - (n + 1) * n * (s / NSsqPXMLsq.array()).sum();
  }
}

void StudentT::backward_param_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double n = in_nodes[0]->value._double;
  double l = in_nodes[1]->value._double;
  double s = in_nodes[2]->value._double;
  Eigen::MatrixXd NSsqPXMLsq = NS2PXML2;
  double adjunct_sum = 1.0;
  if (in_nodes[0]->needs_gradient() or in_nodes[2]->needs_gradient()) {
    adjunct_sum = adjunct.sum();
  }
  if (in_nodes[0]->needs_gradient()) {
    double jacob = adjunct_sum *
        (0.5 * util::polygamma(0, (n + 1) / 2) -
         0.5 * util::polygamma(0, n / 2) - 0.5 / n + 0.5 * std::log(n) +
         std::log(s) + 0.5 * (n + 1) / n);
    in_nodes[0]->back_grad1._double += jacob -
        (adjunct.array() *
         (0.5 * NSsqPXMLsq.array().log() +
          0.5 * (n + 1) * s * s / NSsqPXMLsq.array()))
            .sum();
  }
  if (in_nodes[1]->needs_gradient()) {
    in_nodes[1]->back_grad1._double +=
        (adjunct.array() * (n + 1) * (value._matrix.array() - l) /
         NSsqPXMLsq.array())
            .sum();
  }
  if (in_nodes[2]->needs_gradient()) {
    in_nodes[2]->back_grad1._double += adjunct_sum * n / s -
        (adjunct.array() * (n + 1) * n * (s / NSsqPXMLsq.array())).sum();
  }
}

} // namespace distribution
} // namespace beanmachine
