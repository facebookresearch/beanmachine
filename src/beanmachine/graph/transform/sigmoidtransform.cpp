/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <beanmachine/graph/util.h>
#include "beanmachine/graph/transform/transform.h"

namespace beanmachine {
namespace transform {

// See also https://en.wikipedia.org/wiki/Logit

// y = f(x) = logit(x) = log(x / (1 - x))
// dy/dx = 1 / (x - x^2)
void Sigmoid::operator()(
    const graph::NodeValue& constrained,
    graph::NodeValue& unconstrained) {
  assert(constrained.type.atomic_type == graph::AtomicType::PROBABILITY);
  if (constrained.type.variable_type == graph::VariableType::SCALAR) {
    auto x = constrained._double;
    unconstrained._double = std::log(x / (1 - x));
  } else if (
      constrained.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    auto x = constrained._matrix.array();
    unconstrained._matrix = (x / (1 - x)).log();
  } else {
    throw std::invalid_argument(
        "Sigmoid transformation requires PROBABILITY values.");
  }
}

// x = f^{-1}(y) = expit(y) = 1 / (1 + exp(-y))
// dx/dy = exp(-y) / (1 + exp(-y))^2
void Sigmoid::inverse(
    graph::NodeValue& constrained,
    const graph::NodeValue& unconstrained) {
  assert(constrained.type.atomic_type == graph::AtomicType::PROBABILITY);
  if (constrained.type.variable_type == graph::VariableType::SCALAR) {
    auto y = unconstrained._double;
    constrained._double = 1 / (1 + std::exp(-y));
  } else if (
      constrained.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    auto y = unconstrained._matrix.array();
    constrained._matrix = 1 / (1 + (-y).exp());
  } else {
    throw std::invalid_argument(
        "Sigmoid transformation requires PROBABILITY values.");
  }
}

/*
Return the log of the absolute jacobian determinant:
  log |det(d x / d y)|
:param constrained: the node value x in constrained space
:param unconstrained: the node value y in unconstrained space

for scalar, log |det(d x / d y)|
            = log |exp(-y) / (1 + exp(-y))^2|
            = y - 2 log(1 + exp(y))
for matrix, log |det(d x / d y)|
            = log |prod {y_i - 2 * Log[1 + exp(y_i)]}|
            = sum{y_i - 2 * Log[1 + exp(y_i)]}
*/
double Sigmoid::log_abs_jacobian_determinant(
    const graph::NodeValue& constrained,
    const graph::NodeValue& unconstrained) {
  assert(constrained.type.atomic_type == graph::AtomicType::PROBABILITY);
  if (constrained.type.variable_type == graph::VariableType::SCALAR) {
    auto y = unconstrained._double;
    return y - 2 * util::log1pexp(y);
  } else if (
      constrained.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    /* Because this transformation is applied to each element of the input
    matrix independently, the jabobian is a matrix with only values on the
    diagnonal corresponding to the derivative of that value with respect to the
    corresponding input.  Also, the function is monotonically increasing, so the
    derivative values are positive and so the absolute values of them are
    positive. Therefore, the determinant of the jacobian is the product of the
    diagonal entries, which is the product of the elementwise derivatives.  The
    log of that determinant is the sum of the log of the derivatives. */
    auto y = unconstrained._matrix.array();
    return (y - 2 * util::log1pexp(y).array()).sum();
  } else {
    throw std::invalid_argument(
        "Sigmoid transformation requires PROBABILITY values.");
  }
  return 0;
}

/*
Given the gradient of the joint log prob w.r.t x (constrained), update the value
so that it is taken w.r.t y (unconstrained).
  back_grad = back_grad * dx / dy + d(log |det(d x / d y)|) / dy

:param back_grad: the gradient w.r.t x, a.k.a
:param constrained: the node value x in constrained space
:param unconstrained: the node value y in unconstrained space

Given
x = constrained
y = unconstrained
back_grad = d/dx g(x) = g'(x) // for the joint log_prob function function g

we want (for the first term) d/dy g(x)
   where x = expit(y) = 1 / (1 + exp(-y))
   where expit'(y) = exp(-y) / (1 + exp(-y))^2

d/dy g(x) =
d/dy g(expit(y)) =
g'(expit(y)) expit'(y) =
g'(x) expit'(y) =
back_grad * exp(-y) / (1 + exp(-y))^2

and for the second term:

d(log |det(d x / d y)|) / dy =
    d(sum{y_i - 2 * Log[1 + exp(y_i)]}) / dy =
    {-Tanh[y_i/2]}
*/
void Sigmoid::unconstrained_gradient(
    graph::DoubleMatrix& back_grad,
    const graph::NodeValue& constrained,
    const graph::NodeValue& unconstrained) {
  assert(constrained.type.atomic_type == graph::AtomicType::PROBABILITY);
  if (constrained.type.variable_type == graph::VariableType::SCALAR) {
    auto y = unconstrained._double;
    auto expmy = std::exp(-y); // exp(-y)
    auto dxdy = expmy / std::pow(1 + expmy, 2);
    auto dlddy = -std::tanh(y / 2);
    back_grad = back_grad * dxdy + dlddy;
  } else if (
      constrained.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    auto y = unconstrained._matrix.array();
    auto expmy = (-y).exp(); // exp(-y)
    auto dxdy = expmy / (1 + expmy).pow(2);
    auto dlddy = -(y / 2).tanh();
    back_grad = back_grad.array() * dxdy + dlddy;
  } else {
    throw std::invalid_argument(
        "Sigmoid transformation requires scalar or broadcast matrix values.");
  }
}

} // namespace transform
} // namespace beanmachine
