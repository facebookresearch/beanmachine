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

#include "beanmachine/graph/distribution/normal.h"

namespace beanmachine {
namespace distribution {

using namespace graph;

Normal::Normal(AtomicType sample_type, const std::vector<Node*>& in_nodes)
    : Distribution(DistributionType::NORMAL, sample_type) {
  // a Normal distribution has two parents
  // mean -> real, sigma -> positive real
  if (in_nodes.size() != 2) {
    throw std::invalid_argument(
        "Normal distribution must have exactly two parents");
  }
  if (in_nodes[0]->value.type != graph::AtomicType::REAL or
      in_nodes[1]->value.type != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "Normal parents must be a real number and a positive real number");
  }
  // only real-valued samples are possible
  if (sample_type != AtomicType::REAL) {
    throw std::invalid_argument(
        "Normal distribution produces real number samples");
  }
}

double Normal::_double_sampler(std::mt19937& gen) const {
  std::normal_distribution<double> dist(
      in_nodes[0]->value._value.item().toDouble(), 
      in_nodes[1]->value._value.item().toDouble());
  return dist(gen);
}

// log_prob of a normal: - log(s) -0.5 log(2*pi) - 0.5 (x - m)^2 / s^2
// grad  w.r.t. value x: - (x - m) / s^2
// grad2 w.r.t. value x: - 1 / s^2
// grad  w.r.t. s : -1/s + (x-m)^2 / s^3
// grad2 w.r.t. s : 1/s^2 - 3 (x-m)^2 / s^4
// grad  w.r.t. m : (x - m) / s^2
// grad2 w.r.t. m : -1 / s^2
// First order chain rule: f(g(x))' = f'(g(x)) g'(x),
// - In backward propagation, f'(g(x)) is given by adjunct, the above equation
// computes g'(x). [g is the current function f is the final target]
// - In forward propagation, g'(x) is given by in_nodes[x]->grad1,
// the above equation computes f'(g) [f is the current function g is the input]
torch::Tensor Normal::log_prob(const NodeValue& value) const {
  torch::Tensor m = in_nodes[0]->value._value;
  torch::Tensor s = in_nodes[1]->value._value;
  torch::Tensor result, sum_x, sum_xsq;
  int size;

  if (value.type.variable_type == graph::VariableType::SCALAR) {
    size = 1;
    sum_x = value._value;
    sum_xsq = value._value * value._value;
  } else if (
      value.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    size = static_cast<int>(value._value.numel());
    sum_x = value._value.sum();
    sum_xsq = value._value.norm(2);
  } else {
    throw std::runtime_error(
        "Normal::log_prob applied to invalid variable type");
  }
  result = (-s.log() - 0.5 * std::log(2 * M_PI)) * size -
      0.5 * (sum_xsq - 2 * m * sum_x + m * m * size) / (s * s);
  return result;
}

void Normal::log_prob_iid(
    const graph::NodeValue& value,
    torch::Tensor& log_probs) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  torch::Tensor m = in_nodes[0]->value._value;
  torch::Tensor s = in_nodes[1]->value._value;
  log_probs = (-s.log() - 0.5 * std::log(2 * M_PI)) -
      0.5 * (value._value - m).pow(2) / (s * s);
}

// void Normal::_grad1_log_prob_value(
//     double& grad1,
//     double val,
//     double m,
//     double s_sq) {
//   grad1 += -(val - m) / s_sq;
// };

void Normal::gradient_log_prob_value(
    const NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  // double m = in_nodes[0]->value._value;
  // double s = in_nodes[1]->value._value;
  // double s_sq = s * s;
  // _grad1_log_prob_value(grad1, value._value, m, s_sq);
  // grad2 += -1 / s_sq;

  torch::Tensor lp = log_prob(value);
  torch::Tensor grad = torch::autograd::grad({lp}, {value._value})[0];
  grad1 = grad.item().toDouble();
  grad2 = torch::autograd::grad({grad}, {value._value})[0].item().toDouble();
};

void Normal::gradient_log_prob_param(
    const NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  torch::Tensor x = value._value;
  torch::Tensor m = in_nodes[0]->value._value;
  torch::Tensor s = in_nodes[1]->value._value;
  torch::Tensor s_sq = s * s;
  // gradients of m should be non-zero before computing gradients w.r.t. m
  double m_grad = in_nodes[0]->grad1;
  double m_grad2 = in_nodes[0]->grad2;
  torch::Tensor lp = log_prob(value);
  if (m_grad != 0 or m_grad2 != 0) {
    // torch::Tensor grad_m = (x - m) / s_sq;
    // torch::Tensor grad2_m2 = -1 / s_sq;
    // grad1 += grad_m * m_grad;
    // grad2 += grad2_m2 * m_grad * m_grad + grad_m * m_grad2;
    torch::Tensor grad = torch::autograd::grad(
      {lp}, {m}
    )[0];
    grad1 += grad.item().toDouble();
    grad2 += torch::autograd::grad({grad}, {m})[0].item().toDouble();
  }
  double s_grad = in_nodes[1]->grad1;
  double s_grad2 = in_nodes[1]->grad2;
  if (s_grad != 0 or s_grad2 != 0) {
    // torch::Tensor grad_s = -1 / s + (x - m) * (x - m) / (s * s * s);
    // torch::Tensor grad2_s2 = 1 / s_sq - 3 * (x - m) * (x - m) / (s_sq * s_sq);
    // grad1 += grad_s * s_grad;
    // grad2 += grad2_s2 * s_grad * s_grad + grad_s * s_grad2;
    torch::Tensor grad = torch::autograd::grad(
      {lp}, {s}
    )[0];
    grad1 += grad.item().toDouble();
    grad2 += torch::autograd::grad({grad}, {s})[0].item().toDouble();
  }
}

// void Normal::backward_value(
//     const graph::NodeValue& value,
//     graph::DoubleMatrix& back_grad,
//     double adjunct) const {
//   assert(value.type.variable_type == graph::VariableType::SCALAR);
//   double m = in_nodes[0]->value._value;
//   double s = in_nodes[1]->value._value;
//   double s_sq = s * s;
//   double increment = 0.0;
//   _grad1_log_prob_value(increment, value._value, m, s_sq);
//   back_grad._value += adjunct * increment;
// }

// void Normal::backward_value_iid(
//     const graph::NodeValue& value,
//     graph::DoubleMatrix& back_grad) const {
//   assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
//   double m = in_nodes[0]->value._value;
//   double s = in_nodes[1]->value._value;
//   double s_sq = s * s;
//   back_grad._value -= ((value._value - m) / s_sq);
// }

// void Normal::backward_value_iid(
//     const graph::NodeValue& value,
//     graph::DoubleMatrix& back_grad,
//     torch::Tensor& adjunct) const {
//   assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
//   double m = in_nodes[0]->value._value;
//   double s = in_nodes[1]->value._value;
//   double s_sq = s * s;
//   back_grad._value -=
//       (adjunct * (value._value - m) / s_sq);
// }

// void Normal::backward_param(const graph::NodeValue& value, double adjunct)
//     const {
//   assert(value.type.variable_type == graph::VariableType::SCALAR);
//   double m = in_nodes[0]->value._value;
//   double s = in_nodes[1]->value._value;
//   double s_sq = s * s;
//   double jacob_0 = (value._value - m) / s_sq;

//   if (in_nodes[0]->needs_gradient()) {
//     in_nodes[0]->back_grad1._value += adjunct * jacob_0;
//   }
//   if (in_nodes[1]->needs_gradient()) {
//     in_nodes[1]->back_grad1._value +=
//         adjunct * (-1 / s + jacob_0 * jacob_0 * s);
//   }
// }

// void Normal::backward_param_iid(const graph::NodeValue& value) const {
//   assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
//   double m = in_nodes[0]->value._value;
//   double s = in_nodes[1]->value._value;
//   double s_sq = s * s;

//   int size = static_cast<int>(value._value.numel());
//   double sum_x = value._value.sum().item().toDouble();
//   if (in_nodes[0]->needs_gradient()) {
//     in_nodes[0]->back_grad1._value += sum_x / s_sq - size * m / s_sq;
//   }
//   if (in_nodes[1]->needs_gradient()) {
//     double sum_xsq = value._value.norm(2).item().toDouble();
//     in_nodes[1]->back_grad1._value +=
//         (-size / s + (sum_xsq - 2 * m * sum_x + m * m * size) / (s * s_sq));
//   }
// }

// void Normal::backward_param_iid(
//     const graph::NodeValue& value,
//     torch::Tensor& adjunct) const {
//   assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
//   double m = in_nodes[0]->value._value;
//   double s = in_nodes[1]->value._value;
//   double s_sq = s * s;

//   double sum_x = (value._value * adjunct).sum().item().toDouble();
//   double sum_adjunct = adjunct.sum().item().toDouble();
//   if (in_nodes[0]->needs_gradient()) {
//     in_nodes[0]->back_grad1._value += sum_x / s_sq - sum_adjunct * m / s_sq;
//   }
//   if (in_nodes[1]->needs_gradient()) {
//     double sum_xsq = (value._value.pow(2) * adjunct).sum().item().toDouble();
//     in_nodes[1]->back_grad1._value +=
//         (-sum_adjunct / s +
//          (sum_xsq - 2 * m * sum_x + m * m * sum_adjunct) / (s * s_sq));
//   }
// }

} // namespace distribution
} // namespace beanmachine
