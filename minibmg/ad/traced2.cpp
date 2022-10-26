/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/ad/traced2.h"
#include <functional>
#include <memory>
#include <set>
#include <vector>
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/eval_error.h"
#include "beanmachine/minibmg/node2.h"

namespace beanmachine::minibmg {

double Traced2::as_double() const {
  double value;
  if (is_constant(node, value)) {
    return value;
  }

  throw EvalError("constant value expected but found " + to_string(*this));
}

// We perform some optimizations during construction.
// It might be better to do no optimizations at this point and have a tree
// rewriter that can be reused, but for now this is a simpler approach.
Traced2 operator+(const Traced2& left, const Traced2& right) {
  ScalarNode2p result = std::make_shared<ScalarAddNode2>(left.node, right.node);
  return result;
}

Traced2 operator-(const Traced2& left, const Traced2& right) {
  ScalarNode2p result =
      std::make_shared<ScalarSubtractNode2>(left.node, right.node);
  return result;
}

Traced2 operator-(const Traced2& x) {
  ScalarNode2p result = std::make_shared<ScalarNegateNode2>(x.node);
  return result;
}

Traced2 operator*(const Traced2& left, const Traced2& right) {
  ScalarNode2p result =
      std::make_shared<ScalarMultiplyNode2>(left.node, right.node);
  return result;
}

Traced2 operator/(const Traced2& left, const Traced2& right) {
  ScalarNode2p result =
      std::make_shared<ScalarDivideNode2>(left.node, right.node);
  return result;
}

Traced2 pow(const Traced2& base, const Traced2& exponent) {
  ScalarNode2p result =
      std::make_shared<ScalarPowNode2>(base.node, exponent.node);
  return result;
}

Traced2 exp(const Traced2& x) {
  ScalarNode2p result = std::make_shared<ScalarExpNode2>(x.node);
  return result;
}

Traced2 log(const Traced2& x) {
  ScalarNode2p result = std::make_shared<ScalarLogNode2>(x.node);
  return result;
}

Traced2 atan(const Traced2& x) {
  ScalarNode2p result = std::make_shared<ScalarAtanNode2>(x.node);
  return result;
}

Traced2 lgamma(const Traced2& x) {
  ScalarNode2p result = std::make_shared<ScalarLgammaNode2>(x.node);
  return result;
}

Traced2 polygamma(const int n, const Traced2& x) {
  ScalarNode2p nn = std::make_shared<ScalarConstantNode2>(n);
  ScalarNode2p result = std::make_shared<ScalarPolygammaNode2>(nn, x.node);
  return result;
}

Traced2 if_equal(
    const Traced2& value,
    const Traced2& comparand,
    const Traced2& when_equal,
    const Traced2& when_not_equal) {
  ScalarNode2p result = std::make_shared<ScalarIfEqualNode2>(
      value.node, comparand.node, when_equal.node, when_not_equal.node);
  return result;
}

Traced2 if_less(
    const Traced2& value,
    const Traced2& comparand,
    const Traced2& when_less,
    const Traced2& when_not_less) {
  ScalarNode2p result = std::make_shared<ScalarIfLessNode2>(
      value.node, comparand.node, when_less.node, when_not_less.node);
  return result;
}

bool is_constant(const Traced2& x, double& value) {
  if (auto xnode = dynamic_cast<const ScalarConstantNode2*>(x.node.get())) {
    value = xnode->constant_value;
    return true;
  }
  return false;
}

bool is_constant(const Traced2& x, const double& value) {
  double v;
  return is_constant(x, v) && v == value;
}

std::string to_string(const Traced2& traced) {
  Node2p node = traced.node;
  return to_string(node);
}

} // namespace beanmachine::minibmg
