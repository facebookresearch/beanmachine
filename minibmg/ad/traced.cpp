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

namespace beanmachine::minibmg {

double Traced::as_double() const {
  double value;
  if (is_constant(node, value)) {
    return value;
  }

  throw EvalError("constant value expected but found " + to_string(*this));
}

// We perform some optimizations during construction.
// It might be better to do no optimizations at this point and have a tree
// rewriter that can be reused, but for now this is a simpler approach.
Traced operator+(const Traced& left, const Traced& right) {
  ScalarNodep result = std::make_shared<ScalarAddNode>(left.node, right.node);
  return result;
}

Traced operator-(const Traced& left, const Traced& right) {
  ScalarNodep result =
      std::make_shared<ScalarSubtractNode>(left.node, right.node);
  return result;
}

Traced operator-(const Traced& x) {
  ScalarNodep result = std::make_shared<ScalarNegateNode>(x.node);
  return result;
}

Traced operator*(const Traced& left, const Traced& right) {
  ScalarNodep result =
      std::make_shared<ScalarMultiplyNode>(left.node, right.node);
  return result;
}

Traced operator/(const Traced& left, const Traced& right) {
  ScalarNodep result =
      std::make_shared<ScalarDivideNode>(left.node, right.node);
  return result;
}

Traced pow(const Traced& base, const Traced& exponent) {
  ScalarNodep result =
      std::make_shared<ScalarPowNode>(base.node, exponent.node);
  return result;
}

Traced exp(const Traced& x) {
  ScalarNodep result = std::make_shared<ScalarExpNode>(x.node);
  return result;
}

Traced log(const Traced& x) {
  ScalarNodep result = std::make_shared<ScalarLogNode>(x.node);
  return result;
}

Traced atan(const Traced& x) {
  ScalarNodep result = std::make_shared<ScalarAtanNode>(x.node);
  return result;
}

Traced lgamma(const Traced& x) {
  ScalarNodep result = std::make_shared<ScalarLgammaNode>(x.node);
  return result;
}

Traced polygamma(const int n, const Traced& x) {
  ScalarNodep nn = std::make_shared<ScalarConstantNode>(n);
  ScalarNodep result = std::make_shared<ScalarPolygammaNode>(nn, x.node);
  return result;
}

Traced log1p(const Traced& x) {
  ScalarNodep result = std::make_shared<ScalarLog1pNode>(x.node);
  return result;
}

Traced if_equal(
    const Traced& value,
    const Traced& comparand,
    const Traced& when_equal,
    const Traced& when_not_equal) {
  ScalarNodep result = std::make_shared<ScalarIfEqualNode>(
      value.node, comparand.node, when_equal.node, when_not_equal.node);
  return result;
}

Traced if_less(
    const Traced& value,
    const Traced& comparand,
    const Traced& when_less,
    const Traced& when_not_less) {
  ScalarNodep result = std::make_shared<ScalarIfLessNode>(
      value.node, comparand.node, when_less.node, when_not_less.node);
  return result;
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
