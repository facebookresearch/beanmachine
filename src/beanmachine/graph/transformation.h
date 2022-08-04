/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace beanmachine::graph {

class NodeValue;
struct DoubleMatrix;

enum class TransformType { NONE = 0, LOG = 1 };

class Transformation {
 public:
  Transformation() : transform_type(TransformType::NONE) {}
  explicit Transformation(TransformType transform_type)
      : transform_type(transform_type) {}
  virtual ~Transformation() {}

  /*
  Overload the () to perform the variable transformation y=f(x) from the
  constrained value x to unconstrained y
  :param constrained: the node value x in constrained space
  :param unconstrained: the node value y in unconstrained space
  */
  virtual void operator()(
      const NodeValue& /* constrained */,
      NodeValue& /* unconstrained */) {}
  /*
  Perform the inverse variable transformation x=f^{-1}(y) from the
  unconstrained value y to the original constrained x
  :param constrained: the node value x in constrained space
  :param unconstrained: the node value y in unconstrained space
  */
  virtual void inverse(
      NodeValue& /* constrained */,
      const NodeValue& /* unconstrained */) {}
  /*
  Return the log of the absolute jacobian determinant:
    log |det(d x / d y)|
  :param constrained: the node value x in constrained space
  :param unconstrained: the node value y in unconstrained space
  */
  virtual double log_abs_jacobian_determinant(
      const NodeValue& /* constrained */,
      const NodeValue& /* unconstrained */) {
    return 0;
  }
  /*
  Given the gradient of the joint log prob w.r.t x, update the value so
  that it is taken w.r.t y:
    back_grad = back_grad * dx / dy + d(log |det(d x / d y)|) / dy
  :param back_grad: the gradient w.r.t x
  :param constrained: the node value x in constrained space
  :param unconstrained: the node value y in unconstrained space
  */
  virtual void unconstrained_gradient(
      DoubleMatrix& /* back_grad */,
      const NodeValue& /* constrained */,
      const NodeValue& /* unconstrained */) {}

  TransformType transform_type;
};

} // namespace beanmachine::graph
