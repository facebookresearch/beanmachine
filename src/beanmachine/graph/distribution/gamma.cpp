/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <random>
#include <string>

#include "beanmachine/graph/distribution/gamma.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace distribution {

using namespace graph;

Gamma::Gamma(AtomicType sample_type, const std::vector<Node*>& in_nodes)
    : Distribution(DistributionType::GAMMA, sample_type) {
  // a Gamma distribution has two parents:
  // shape -> positive real; rate -> positive real
  if (sample_type != AtomicType::POS_REAL) {
    throw std::invalid_argument("Gamma produces positive real samples");
  }
  if (in_nodes.size() != 2) {
    throw std::invalid_argument(
        "Gamma distribution must have exactly two parents");
  }
  if (in_nodes[0]->value.type != AtomicType::POS_REAL or
      in_nodes[1]->value.type != AtomicType::POS_REAL) {
    throw std::invalid_argument("Gamma parents must be positive real-valued");
  }
}

double Gamma::_double_sampler(std::mt19937& gen) const {
  std::gamma_distribution<double> dist(
      in_nodes[0]->value._double, 1 / in_nodes[1]->value._double);
  return dist(gen);
}

// Note: log_prob(x | a, b) = a * log(b) - log G(a) + (a - 1) * log(x) - b * x
// grad1 w.r.t. x = (a - 1) / x - b
// grad2 w.r.t. x = (1 - a) / x^2
// grad1 w.r.t. params = [log(b) - digamma(a) + log(x)] * a'
//                     + (a / b - x) * b'
// grad2 w.r.t. params = - polygamma(1, a) * (a')^2
//                       + [log(b) - digamma(a) + log(x)] * a''
//                       - a / b^2 * (b')^2 + (a / b - x) * b''
// First order chain rule: f(g(x))' = f'(g(x)) g'(x),
// - In backward propagation, f'(g(x)) is given by adjunct, the above equation
// computes g'(x). [g is the current function f is the final target]
// - In forward propagation, g'(x) is given by in_nodes[x]->grad1,
// the above equation computes f'(g) [f is the current function g is the input]
double Gamma::log_prob(const graph::NodeValue& value) const {
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;

  double result = param_a * std::log(param_b) - lgamma(param_a);
  if (value.type.variable_type == graph::VariableType::SCALAR) {
    result +=
        (param_a - 1.0) * std::log(value._double) - param_b * value._double;
  } else if (
      value.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    result *= value._matrix.size();
    result += (param_a - 1.0) * value._matrix.array().log().sum() -
        param_b * value._matrix.sum();
  } else {
    throw std::runtime_error(
        "Gamma::log_prob applied to invalid variable type");
  }
  return result;
}

void Gamma::log_prob_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& log_probs) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  double result = param_a * std::log(param_b) - lgamma(param_a);
  log_probs = result + (param_a - 1.0) * value._matrix.array().log() -
      param_b * value._matrix.array();
}

void Gamma::_grad1_log_prob_value(
    double& grad1,
    double x,
    double param_a,
    double param_b) {
  grad1 += (param_a - 1.0) / x - param_b;
}

void Gamma::gradient_log_prob_value(
    const graph::NodeValue& value,
    double& grad1,
    double& grad2) const {
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  _grad1_log_prob_value(grad1, value._double, param_a, param_b);
  grad2 += (1.0 - param_a) / (value._double * value._double);
}

void Gamma::gradient_log_prob_param(
    const graph::NodeValue& value,
    double& grad1,
    double& grad2) const {
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  double digamma_a = util::polygamma(0, param_a); // digamma(a)
  double poly1_a = util::polygamma(1, param_a); // polygamma(1, a)
  // 1st order derivatives
  double grad_a = std::log(param_b) - digamma_a + std::log(value._double);
  double grad_b = param_a / param_b - value._double;
  // 2nd order derivatives
  double grad2_a2 = -poly1_a;
  double grad2_b2 = -param_a / (param_b * param_b);
  // combine with chain rule
  grad1 += grad_a * in_nodes[0]->grad1 + grad_b * in_nodes[1]->grad1;
  grad2 += grad_a * in_nodes[0]->grad2 + grad_b * in_nodes[1]->grad2 +
      grad2_a2 * in_nodes[0]->grad1 * in_nodes[0]->grad1 +
      grad2_b2 * in_nodes[1]->grad1 * in_nodes[1]->grad1;
}

void Gamma::backward_value(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    double adjunct) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  double increment = 0.0;
  _grad1_log_prob_value(increment, value._double, param_a, param_b);
  back_grad._double += adjunct * increment;
}

void Gamma::backward_value_iid(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  back_grad._matrix +=
      ((param_a - 1.0) / value._matrix.array() - param_b).matrix();
}

void Gamma::backward_value_iid(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  back_grad._matrix +=
      (adjunct.array() * ((param_a - 1.0) / value._matrix.array() - param_b))
          .matrix();
}

void Gamma::backward_param(const graph::NodeValue& value, double adjunct)
    const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  if (in_nodes[0]->needs_gradient()) {
    double digamma_a = util::polygamma(0, param_a); // digamma(a)
    double jacob = std::log(param_b) - digamma_a + std::log(value._double);
    in_nodes[0]->back_grad1._double += adjunct * jacob;
  }
  if (in_nodes[1]->needs_gradient()) {
    double jacob = param_a / param_b - value._double;
    in_nodes[1]->back_grad1._double += adjunct * jacob;
  }
}

void Gamma::backward_param_iid(const graph::NodeValue& value) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  int size = static_cast<int>(value._matrix.size());
  if (in_nodes[0]->needs_gradient()) {
    double digamma_a = util::polygamma(0, param_a); // digamma(a)
    in_nodes[0]->back_grad1._double += size * (std::log(param_b) - digamma_a) +
        value._matrix.array().log().sum();
  }
  if (in_nodes[1]->needs_gradient()) {
    in_nodes[1]->back_grad1._double +=
        size * (param_a / param_b) - value._matrix.array().sum();
  }
}

void Gamma::backward_param_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  double adjunct_sum = 1.0;
  if (in_nodes[0]->needs_gradient() or in_nodes[1]->needs_gradient()) {
    adjunct_sum = adjunct.sum();
  }
  if (in_nodes[0]->needs_gradient()) {
    double digamma_a = util::polygamma(0, param_a); // digamma(a)
    in_nodes[0]->back_grad1._double +=
        adjunct_sum * (std::log(param_b) - digamma_a) +
        (value._matrix.array().log() * adjunct.array()).sum();
  }
  if (in_nodes[1]->needs_gradient()) {
    in_nodes[1]->back_grad1._double += adjunct_sum * (param_a / param_b) -
        (value._matrix.array() * adjunct.array()).sum();
  }
}

} // namespace distribution
} // namespace beanmachine
