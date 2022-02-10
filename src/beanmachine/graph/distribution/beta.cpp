/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include "beanmachine/graph/distribution/beta.h"
#include "beanmachine/graph/util.h"

using namespace torch::indexing;

namespace beanmachine {
namespace distribution {

Beta::Beta(
    graph::AtomicType sample_type,
    const std::vector<graph::Node*>& in_nodes)
    : Distribution(graph::DistributionType::BETA, sample_type) {
  // a Beta has two parents which are real numbers and it outputs a probability
  if (sample_type != graph::AtomicType::PROBABILITY) {
    throw std::invalid_argument("Beta produces probability samples");
  }
  if (in_nodes.size() != 2) {
    throw std::invalid_argument(
        "Beta distribution must have exactly two parents");
  }
  if (in_nodes[0]->value.type != graph::AtomicType::POS_REAL or
      in_nodes[1]->value.type != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument("Beta parents must be positive real-valued");
  }
}

double Beta::_double_sampler(std::mt19937& gen) const {
  return util::sample_beta(
      gen, in_nodes[0]->value._double, in_nodes[1]->value._double);
}

double Beta::log_prob(const graph::NodeValue& value) const {
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  double ret_val = 0.0;
  auto update_logprob = [&](double val) {
    ret_val += (param_a - 1) * log(val) + (param_b - 1) * log(1 - val);
  };
  if (value.type.variable_type == graph::VariableType::SCALAR) {
    update_logprob(value._double);
    ret_val += lgamma(param_a + param_b) - lgamma(param_a) - lgamma(param_b);
    return ret_val;
  }

  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  assert(value.type.rows * value.type.cols > 1);
  const uint size = static_cast<uint>(value._matrix.numel());
  auto value_a = value._matrix.accessor<double, 2>();
  for (uint i = 0; i < value_a.size(0); i++) {
    for (uint j = 0; i < value_a.size(1); j++) {
      update_logprob(value_a[i][j]);
      // update_logprob(*(value._matrix.data() + i));
    }
  }
  ret_val +=
      size * (lgamma(param_a + param_b) - lgamma(param_a) - lgamma(param_b));
  return ret_val;
}

void Beta::log_prob_iid(
    const graph::NodeValue& value,
    torch::Tensor& log_probs) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  double result = lgamma(param_a + param_b) - lgamma(param_a) - lgamma(param_b);
  log_probs = result + (param_a - 1) * value._matrix.log() +
      (param_b - 1) * (1 - value._matrix).log();
}

// Note log_prob(x | a, b) = (a-1) log(x) + (b-1) log(1-x) + log G(a+b) - log
// G(a) - log G(b) grad1 w.r.t. x =  (a-1) / x - (b-1) / (1-x) grad2 w.r.t. x =
// - (a-1) / x^2 - (b-1) / (1-x)^2 grad1 w.r.t. params = (log(x) + digamma(a+b)
// - digamma(a)) a'
//                        + (log(1-x) + digamma(a+b) - digamma(b)) b'
// grad2 w.r.t. params =
//   (polygamma(1, a+b) - polygamma(1, a)) a'^2 + (log(x) + digamma(a+b) -
//   digamma(a)) a'' (polygamma(1, a+b) - polygamma(1, b)) b'^2 + (log(1-x) +
//   digamma(a+b) - digamma(b)) b''
// First order chain rule: f(g(x))' = f'(g(x)) g'(x),
// - In backward propagation, f'(g(x)) is given by adjunct, the above equation
// computes g'(x). [g is the current function f is the final target]
// - In forward propagation, g'(x) is given by in_nodes[x]->grad1,
// the above equation computes f'(g) [f is the current function g is the input]

void Beta::_grad1_log_prob_value(
    double& grad1,
    double x,
    double param_a,
    double param_b) {
  grad1 += (param_a - 1) / x - (param_b - 1) / (1 - x);
}

void Beta::gradient_log_prob_value(
    const graph::NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double x = value._double;
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  _grad1_log_prob_value(grad1, x, param_a, param_b);
  grad2 += -(param_a - 1) / (x * x) - (param_b - 1) / ((1 - x) * (1 - x));
}

void Beta::gradient_log_prob_param(
    const graph::NodeValue& value,
    double& grad1,
    double& grad2) const {
  torch::Tensor jacobian = torch::empty({1, 2});
  torch::Tensor hessian = torch::empty({2, 2});
  compute_jacobian_hessian(value, jacobian, hessian);
  forward_gradient_scalarops(jacobian, hessian, grad1, grad2);
}

void Beta::compute_jacobian_hessian(
    const graph::NodeValue& value,
    torch::Tensor& jacobian,
    torch::Tensor& hessian) const {
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  double digamma_a_p_b = util::polygamma(0, param_a + param_b); // digamma(a+b)
  double digamma_diff_a = digamma_a_p_b - util::polygamma(0, param_a);
  double digamma_diff_b = digamma_a_p_b - util::polygamma(0, param_b);
  double poly1_a_p_b =
      util::polygamma(1, param_a + param_b); // polygamma(1, a+b)

  hessian.index_put_({0, 0}, poly1_a_p_b - util::polygamma(1, param_a));
  hessian.index_put_({1, 0}, poly1_a_p_b);
  hessian.index_put_({0, 1}, poly1_a_p_b);
  hessian.index_put_({1, 1}, poly1_a_p_b - util::polygamma(1, param_b));

  if (value.type.variable_type == graph::VariableType::SCALAR) {
    jacobian.index_put_({0, 0}, std::log(value._double) + digamma_diff_a);
    jacobian.index_put_({0, 1}, std::log(1 - value._double) + digamma_diff_b);
    return;
  }

  uint size = static_cast<uint>(value._matrix.numel());
  assert(size > 1);
  jacobian[0][0] = size * digamma_diff_a;
  jacobian[0][1] = size * digamma_diff_b;
  for (uint i = 0; i < size; i++) {
    jacobian[0][0] += std::log(value._matrix[i].item().toDouble());
    jacobian[0][1] += std::log(1 - value._matrix[i].item().toDouble());
  }
  hessian *= static_cast<double>(size);
}

void Beta::backward_value(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    double adjunct) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double x = value._double;
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  double increment = 0.0;
  _grad1_log_prob_value(increment, x, param_a, param_b);
  back_grad._double += adjunct * increment;
}

void Beta::backward_value_iid(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  back_grad._matrix += ((param_a - 1) / value._matrix -
                        (param_b - 1) / (1 - value._matrix));
}

void Beta::backward_value_iid(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    torch::Tensor& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  back_grad._matrix += (adjunct *
                        ((param_a - 1) / value._matrix -
                         (param_b - 1) / (1 - value._matrix)));
}

void Beta::backward_param(const graph::NodeValue& value, double adjunct) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double x = value._double;
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  double digamma_a_p_b = util::polygamma(0, param_a + param_b);
  if (in_nodes[0]->needs_gradient()) {
    double jacob = std::log(x) + digamma_a_p_b - util::polygamma(0, param_a);
    in_nodes[0]->back_grad1._double += adjunct * jacob;
  }
  if (in_nodes[1]->needs_gradient()) {
    double jacob =
        std::log(1 - x) + digamma_a_p_b - util::polygamma(0, param_b);
    in_nodes[1]->back_grad1._double += adjunct * jacob;
  }
}

void Beta::backward_param_iid(const graph::NodeValue& value) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  double digamma_a_p_b = util::polygamma(0, param_a + param_b);
  int size = static_cast<int>(value._matrix.numel());
  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1._double += value._matrix.log().sum().item().toDouble() +
        size * (digamma_a_p_b - util::polygamma(0, param_a));
  }
  if (in_nodes[1]->needs_gradient()) {
    in_nodes[1]->back_grad1._double += (1 - value._matrix).log().sum().item().toDouble() +
        size * (digamma_a_p_b - util::polygamma(0, param_b));
  }
}

void Beta::backward_param_iid(
    const graph::NodeValue& value,
    torch::Tensor& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  double digamma_a_p_b = util::polygamma(0, param_a + param_b);
  double adjunct_sum = 1.0;
  if (in_nodes[0]->needs_gradient() or in_nodes[1]->needs_gradient()) {
    adjunct_sum = adjunct.sum().item().toDouble();
  }
  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1._double +=
        (adjunct * value._matrix.log()).sum().item().toDouble() +
        adjunct_sum * (digamma_a_p_b - util::polygamma(0, param_a));
  }
  if (in_nodes[1]->needs_gradient()) {
    in_nodes[1]->back_grad1._double +=
        (adjunct * (1 - value._matrix).log()).sum().item().toDouble() +
        adjunct_sum * (digamma_a_p_b - util::polygamma(0, param_b));
  }
}

} // namespace distribution
} // namespace beanmachine
