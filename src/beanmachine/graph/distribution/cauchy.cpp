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

#include "beanmachine/graph/distribution/cauchy.h"

namespace beanmachine {
namespace distribution {

using namespace graph;

Cauchy::Cauchy(AtomicType sample_type, const std::vector<Node*>& in_nodes)
    : Distribution(DistributionType::CAUCHY, sample_type) {
  // a Cauchy distribution has two parents - a location and a scale which are
  // positive reals
  if (in_nodes.size() != 2) {
    throw std::invalid_argument(
        "Cauchy distribution must have exactly two parents");
  }
  auto x0 = in_nodes[0];
  auto s = in_nodes[1];
  if (x0->value.type != graph::AtomicType::REAL) {
    throw std::invalid_argument("Cauchy first parent (location) must be real");
  }
  if (s->value.type != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "Cauchy second parent (scale) must be a positive real number");
  }
  // only real-valued samples are possible
  if (sample_type != AtomicType::REAL) {
    throw std::invalid_argument(
        "Cauchy distribution produces real number samples");
  }
}

double Cauchy::_double_sampler(std::mt19937& gen) const {
  // the CDF of a standard Cauchy is  F(x) = (1/pi) arctan(x)
  // therefore we will sample w ~ uniformly [-π/2, π/2] and compute tan(w).
  // Finally we will multiply the scale "s" and add the location "x0" to get the
  // required value
  double x0 = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  std::uniform_real_distribution<double> dist(-M_PI_2, M_PI_2);
  return x0 + s * std::tan(dist(gen));
}

// The PDF of a cauchy distribution is
// PDF = 1/(π * s * (1 + ((x - x0)/s)^2))
// where x0 is the offset parameter, and
// s is the scale parameter.
// (See https://en.wikipedia.org/wiki/Cauchy_distribution)
// log(PDF) = -log(π * s) - log(1 + ((x - x0)/s)^2)
double Cauchy::log_prob(const NodeValue& value) const {
  double x0 = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  int size;
  if (value.type.variable_type == graph::VariableType::SCALAR) {
    // log(PDF) = -log(π * s) - log(1 + ((x - x0)/s)^2)
    auto x = value._double;
    auto scaledX = (x - x0) / s;
    auto scaledX2 = scaledX * scaledX;
    return (-log(M_PI * s)) - log1p(scaledX2);
  } else if (
      value.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    size = static_cast<int>(value._matrix.size());
    auto x = value._matrix.array();
    auto scaledX = (x - x0) / s;
    auto scaledX2 = scaledX.pow(2);
    return (-log(M_PI * s)) * size - scaledX2.log1p().sum();
  } else {
    throw std::runtime_error(
        "Cauchy::log_prob applied to invalid variable type");
  }
}

void Cauchy::log_prob_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& log_probs) const {
  // log(PDF) = -log(π * s) - log(1 + ((x - x0)/s)^2)
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double x0 = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  auto x = value._matrix.array();
  auto scaledX = (x - x0) / s;
  auto scaledX2 = scaledX.pow(2);
  log_probs = (-log(M_PI * s)) - scaledX2.log1p();
}

void Cauchy::gradient_log_prob_value(
    const NodeValue& value,
    double& grad1,
    double& grad2) const {
  // log(PDF) = -log(π * s) - log(1 + ((x - x0)/s)^2)
  // Using https://www.wolframalpha.com/
  // D[log(PDF[CauchyDistribution[x0, s], x]), x]
  //                 = -2(x - x0) / (s^2 + (x - x0)^2)
  // D[D[log(PDF[CauchyDistribution[x0, s], x]), x], x]
  //                 = (2 (-s^2 + (x - x0)^2))/(s^2 + (x - x0)^2)^2
  //
  // We do not need to use the chain rule as the incoming value is the
  // value at which we are taking the derivative (its first derivative is
  // one and its second derivative is zero).
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double x = value._double;
  double x0 = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  auto t1 = x - x0; // x - x0
  auto t2 = t1 * t1; // (x - x0)^2
  auto s2 = s * s; // s^2
  auto t3 = s2 + t2; // s^2 + (x - x0)^2
  auto d1 = -2 * t1 / t3; // (-2 (x - x0)) / (s^2 + (x - x0)^2)
  auto t4 = t3 * t3; // (s^2 + (x - x0)^2)^2
  auto d2 = 2 * (-s2 + t2) / t4; // (2 (-s^2 + (x - x0)^2))/(s^2 + (x - x0)^2)^2
  grad1 += d1;
  grad2 += d2;
}

void Cauchy::gradient_log_prob_param(
    const NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  // Note: First order chain rule:
  // d/dx[f(g(x))]
  //     = f’(g(x)) g’(x)
  // Second order chain rule:
  // d^2/dx^2[f(g(x))]
  //     = d/dx[f’(g(x)) g’(x)] // first order chain rule
  //     = d/dx[f’(g(x))] g’(x) // product rule
  //       + d/dx[g’(x)] f’(g(x))
  //     = f’’(g(x)) g’(x) g’(x) // first order chain rule
  //       + g’’(x) f’(g(x))
  //     = f’’(g(x)) g’(x) g’(x) + g’’(x) f’(g(x))
  // Here f is log(PDF), g is the value of the parameter to the function
  // being differentiated, and g' and g'' are the incoming gradients of
  // the value.

  double x = value._double;
  double x0 = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;

  double x0_grad1 = in_nodes[0]->grad1;
  double x0_grad2 = in_nodes[0]->grad2;
  double s_grad1 = in_nodes[1]->grad1;
  double s_grad2 = in_nodes[1]->grad2;
  if (x0_grad1 == 0 && x0_grad2 == 0 && s_grad1 == 0 && s_grad2 == 0) {
    return;
  }

  double t1 = x - x0; // (x - x0)
  double t2 = t1 * t1; // (x - x0)^2
  double t3 = s * s; // s^2
  double t4 = t3 + t2; // s^2 + (x - x0)^2
  if (x0_grad1 != 0 or x0_grad2 != 0) {
    // Using https://www.wolframalpha.com/
    // D[log(PDF[CauchyDistribution[x0, s], x]), x0]
    //           = (2 (x - x0))/(s^2 + (x - x0)^2)
    // D[D[log(PDF[CauchyDistribution[x0, s], x]), x0], x0]
    //           = (2 (-s^2 + (x - x0)^2))/(s^2 + (x - x0)^2)^2
    double d1 = 2 * t1 / t4; // (2 (x - x0))/(s^2 + (x - x0)^2)
    double d2 = 2 * (t2 - t3) /
        (t4 * t4); // (2 ((x - x0)^2) - s^2)/(s^2 + (x - x0)^2)^2
    // Use the first and second order chain rules
    grad1 += d1 * x0_grad1;
    grad2 += d2 * x0_grad1 * x0_grad1 + x0_grad2 * d1;
  }

  if (s_grad1 != 0 or s_grad2 != 0) {
    // D[log(PDF[CauchyDistribution[x0, s], x]), s]
    //           = (-s^2 + (x - x0)^2)/(s (s^2 + (x - x0)^2))
    double d1 =
        (t2 - t3) / (s * t4); // ((x - x0)^2 - s^2)/(s (s^2 + (x - x0)^2))
    // D[D[log(PDF[CauchyDistribution[x0, s], x]), s], s]
    //           = (s^4 - 4 s^2 (x - x0)^2 - (x - x0)^4) /
    //             (s^2 (s^2 + (x - x0)^2)^2)
    double s4 = t3 * t3; // s^4
    double t5 = t2 * t2; // (x - x0)^4
    double d2 =
        (s4 - 4 * t3 * t2 - t5) / // (s^4 - 4 s^2 (x - x0)^2 - (x - x0)^4) /
        (t3 * t4 * t4); // (s^2 (s^2 + (x - x0)^2)^2)
    // Use the first and second order chain rules
    grad1 += d1 * s_grad1;
    grad2 += d2 * s_grad1 * s_grad1 + s_grad2 * d1;
  }
}

void Cauchy::backward_value(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    double adjunct) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double x = value._double;
  double x0 = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  // D[log(PDF[CauchyDistribution[x0, s], x]), x]
  //             = -2(x - x0) / (s^2 + (x - x0)^2)
  double t1 = x - x0; // (x - x0)
  double d = s * s + t1 * t1; // (s^2 + (x - x0)^2)
  double grad = -2 * t1 / d; // -2(x - x0) / (s^2 + (x - x0)^2)
  back_grad += adjunct * grad;
}

void Cauchy::backward_value_iid(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  auto x = value._matrix.array();
  double x0 = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  // D[log(PDF[CauchyDistribution[x0, s], x]), x]
  //             = -2(x - x0) / (s^2 + (x - x0)^2)
  auto t1 = x - x0; // (x - x0)
  auto d = s * s + t1 * t1; // (s^2 + (x - x0)^2)
  auto grad = -2 * t1 / d; // -2(x - x0) / (s^2 + (x - x0)^2)
  back_grad += grad;
}

void Cauchy::backward_value_iid(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  auto x = value._matrix.array();
  double x0 = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  // D[log(PDF[CauchyDistribution[x0, s], x]), x]
  //             = -2(x - x0) / (s^2 + (x - x0)^2)
  auto t1 = x - x0;
  auto d = s * s + t1 * t1;
  auto grad = -2 * t1 / d;
  back_grad += adjunct.array() * grad;
}

void Cauchy::backward_param(const graph::NodeValue& value, double adjunct)
    const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  if (adjunct == 0) {
    return;
  }
  auto x = value._double;
  auto x0_node = in_nodes[0];
  double x0 = x0_node->value._double;
  auto s_node = in_nodes[1];
  double s = s_node->value._double;
  auto t1 = x - x0; // x - x0
  auto s2 = s * s; // s^2
  auto t2 = t1 * t1; // (x - x0)^2
  auto t3 = s2 + t2; // (s^2 + (x - x0)^2)
  if (x0_node->needs_gradient()) {
    // D[log(PDF[CauchyDistribution[x0, s], x]), x0]
    //           = (2 (x - x0))/(s^2 + (x - x0)^2)
    auto derivative1 = 2 * t1 / t3; // (2 (x - x0))/(s^2 + (x - x0)^2)
    x0_node->back_grad1 += adjunct * derivative1;
  }
  if (s_node->needs_gradient()) {
    // D[log(PDF[CauchyDistribution[x0, s], x]), s]
    //           = (-s^2 + (x - x0)^2)/(s (s^2 + (x - x0)^2))
    auto t4 = t2 - s2; // ((x - x0)^2 - s^2)
    auto derivative1 =
        t4 / (s * t3); // ((x - x0)^2 - s^2)/(s (s^2 + (x - x0)^2))
    s_node->back_grad1 += adjunct * derivative1;
  }
}

void Cauchy::backward_param_iid(const graph::NodeValue& value) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  auto x = value._matrix.array();
  auto x0_node = in_nodes[0];
  double x0 = x0_node->value._double;
  auto s_node = in_nodes[1];
  double s = s_node->value._double;
  auto t1 = x - x0; // x - x0
  auto s2 = s * s; // s^2
  auto t2 = t1 * t1; // (x - x0)^2
  auto t3 = s2 + t2; // (s^2 + (x - x0)^2)
  if (x0_node->needs_gradient()) {
    // D[log(PDF[CauchyDistribution[x0, s], x]), x0]
    //           = (2 (x - x0))/(s^2 + (x - x0)^2)
    auto derivative1 = 2 * t1 / t3; // (2 (x - x0))/(s^2 + (x - x0)^2)
    x0_node->back_grad1 += derivative1;
  }
  if (s_node->needs_gradient()) {
    // D[log(PDF[CauchyDistribution[x0, s], x]), s]
    //           = (-s^2 + (x - x0)^2)/(s (s^2 + (x - x0)^2))
    auto t4 = t2 - s2; // ((x - x0)^2 - s^2)
    auto derivative1 =
        t4 / (s * t3); // ((x - x0)^2 - s^2)/(s (s^2 + (x - x0)^2))
    s_node->back_grad1 += derivative1;
  }
}

void Cauchy::backward_param_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  auto x = value._matrix.array();
  auto x0_node = in_nodes[0];
  double x0 = x0_node->value._double;
  auto s_node = in_nodes[1];
  double s = s_node->value._double;
  // D[log(PDF), x0] = (2 (x - x0))/(s^2 + (x - x0)^2)
  // D[log(PDF), s] = ((x - x0)^2 - s^2)/(s (s^2 + (x - x0)^2))
  auto t1 = x - x0; // x - x0
  auto s2 = s * s; // s^2
  auto t2 = t1 * t1; // (x - x0)^2
  auto t3 = s2 + t2; // (s^2 + (x - x0)^2)
  auto adjunct_array = adjunct.array();
  if (x0_node->needs_gradient()) {
    auto derivative1 = 2 * t1 / t3; // (2 (x - x0))/(s^2 + (x - x0)^2)
    x0_node->back_grad1 += adjunct_array * derivative1;
  }
  if (s_node->needs_gradient()) {
    auto t4 = t2 - s2; // ((x - x0)^2 - s^2)
    auto derivative1 =
        t4 / (s * t3); // ((x - x0)^2 - s^2)/(s (s^2 + (x - x0)^2))
    s_node->back_grad1 += adjunct_array * derivative1;
  }
}

} // namespace distribution
} // namespace beanmachine
