/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/ad/traced.h"
#include <functional>
#include <memory>
#include <set>
#include <vector>
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/eval_error.h"
#include "beanmachine/minibmg/node.h"
#include "beanmachine/minibmg/rewriters/constant_fold.h"

namespace beanmachine::minibmg {

double Traced::as_double() const {
  double value;
  if (is_constant(node, value)) {
    return value;
  }

  throw EvalError("constant value expected but found " + to_string(*this));
}

// We perform few optimizations during construction - only constant folding.
Traced operator+(const Traced& left, const Traced& right) {
  ScalarNodep result =
      std::make_shared<const ScalarAddNode>(left.node, right.node);
  return constant_fold(result);
}

Traced operator-(const Traced& left, const Traced& right) {
  ScalarNodep result =
      std::make_shared<const ScalarSubtractNode>(left.node, right.node);
  return constant_fold(result);
}

Traced operator-(const Traced& x) {
  ScalarNodep result = std::make_shared<const ScalarNegateNode>(x.node);
  return constant_fold(result);
}

Traced operator*(const Traced& left, const Traced& right) {
  ScalarNodep result =
      std::make_shared<const ScalarMultiplyNode>(left.node, right.node);
  return constant_fold(result);
}

Traced operator/(const Traced& left, const Traced& right) {
  ScalarNodep result =
      std::make_shared<const ScalarDivideNode>(left.node, right.node);
  return constant_fold(result);
}

Traced pow(const Traced& base, const Traced& exponent) {
  ScalarNodep result =
      std::make_shared<const ScalarPowNode>(base.node, exponent.node);
  return constant_fold(result);
}

Traced exp(const Traced& x) {
  ScalarNodep result = std::make_shared<const ScalarExpNode>(x.node);
  return constant_fold(result);
}

Traced log(const Traced& x) {
  ScalarNodep result = std::make_shared<const ScalarLogNode>(x.node);
  return constant_fold(result);
}

Traced atan(const Traced& x) {
  ScalarNodep result = std::make_shared<const ScalarAtanNode>(x.node);
  return constant_fold(result);
}

Traced lgamma(const Traced& x) {
  ScalarNodep result = std::make_shared<const ScalarLgammaNode>(x.node);
  return constant_fold(result);
}

Traced polygamma(const int n, const Traced& x) {
  ScalarNodep nn = std::make_shared<const ScalarConstantNode>(n);
  ScalarNodep result = std::make_shared<const ScalarPolygammaNode>(nn, x.node);
  return constant_fold(result);
}

Traced log1p(const Traced& x) {
  ScalarNodep result = std::make_shared<const ScalarLog1pNode>(x.node);
  return constant_fold(result);
}

Traced if_equal(
    const Traced& value,
    const Traced& comparand,
    const Traced& when_equal,
    const Traced& when_not_equal) {
  ScalarNodep result = std::make_shared<const ScalarIfEqualNode>(
      value.node, comparand.node, when_equal.node, when_not_equal.node);
  return constant_fold(result);
}

Traced if_less(
    const Traced& value,
    const Traced& comparand,
    const Traced& when_less,
    const Traced& when_not_less) {
  ScalarNodep result = std::make_shared<const ScalarIfLessNode>(
      value.node, comparand.node, when_less.node, when_not_less.node);
  return constant_fold(result);
}

bool is_constant(const Traced& x, double& value) {
  if (auto xnode = dynamic_cast<const ScalarConstantNode*>(x.node.get())) {
    value = xnode->constant_value;
    return true;
  }
  return false;
}

bool is_constant(const Traced& x, const double& value) {
  double v;
  return is_constant(x, v) && v == value;
}

std::string to_string(const Traced& traced) {
  Nodep node = traced.node;
  return to_string(node);
}

} // namespace beanmachine::minibmg
