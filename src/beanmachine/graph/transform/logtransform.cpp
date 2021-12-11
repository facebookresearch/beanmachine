/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/transform/transform.h"

namespace beanmachine {
namespace transform {

// y = f(x) = log(x)
void Log::operator()(
    const graph::NodeValue& constrained,
    graph::NodeValue& unconstrained) {
  assert(constrained.type.atomic_type == graph::AtomicType::POS_REAL);
  if (constrained.type.variable_type == graph::VariableType::SCALAR) {
    unconstrained._double = std::log(constrained._double);
  } else if (
      constrained.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    unconstrained._matrix = constrained._matrix.array().log();
  } else {
    throw std::invalid_argument("Log transformation requires POS_REAL values.");
  }
}

// x = f^{-1}(y) = exp(y)
void Log::inverse(
    graph::NodeValue& constrained,
    const graph::NodeValue& unconstrained) {
  assert(constrained.type.atomic_type == graph::AtomicType::POS_REAL);
  if (constrained.type.variable_type == graph::VariableType::SCALAR) {
    constrained._double = std::exp(unconstrained._double);
  } else if (
      constrained.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    constrained._matrix = unconstrained._matrix.array().exp();
  } else {
    throw std::invalid_argument("Log transformation requires POS_REAL values.");
  }
}

// for scalar, log |det(d x / d y)| = log |exp(y)| = y
// for matrix, log |det(d x / d y)| = log |prod {exp(y_i)}| = sum{y_i}
double Log::log_abs_jacobian_determinant(
    const graph::NodeValue& constrained,
    const graph::NodeValue& unconstrained) {
  assert(constrained.type.atomic_type == graph::AtomicType::POS_REAL);
  if (constrained.type.variable_type == graph::VariableType::SCALAR) {
    return unconstrained._double;
  } else if (
      constrained.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    return unconstrained._matrix.sum();
  } else {
    throw std::invalid_argument("Log transformation requires POS_REAL values.");
  }
  return 0;
}

// back_grad = back_grad * dx / dy + d(log |det(d x / d y)|) / dy, where
// dx / dy = exp(y) = x
// d(log |det(d x / d y)|) / dy =  1
void Log::unconstrained_gradient(
    graph::DoubleMatrix& back_grad,
    const graph::NodeValue& constrained,
    const graph::NodeValue& /* unconstrained */) {
  assert(constrained.type.atomic_type == graph::AtomicType::POS_REAL);
  if (constrained.type.variable_type == graph::VariableType::SCALAR) {
    back_grad._double = back_grad._double * constrained._double + 1.0;
  } else if (
      constrained.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    back_grad._matrix =
        back_grad._matrix.array() * constrained._matrix.array() + 1.0;
  } else {
    throw std::invalid_argument("Log transformation requires POS_REAL values.");
  }
}

} // namespace transform
} // namespace beanmachine
