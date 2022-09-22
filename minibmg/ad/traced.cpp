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
#include "beanmachine/minibmg/eval.h"

namespace {

using namespace beanmachine::minibmg;

inline Nodep make_operator(
    Operator op,
    std::vector<Nodep> in_nodes,
    Type type = Type::REAL) {
  return std::make_shared<OperatorNode>(in_nodes, op, type);
}

} // namespace

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
  double left_value, right_value;
  auto left_is_constant = is_constant(left, left_value);
  auto right_is_constant = is_constant(right, right_value);
  if (left_is_constant && right_is_constant) {
    return left_value + right_value;
  }
  if (left_is_constant && left_value == 0) {
    return right;
  }
  if (right_is_constant && right_value == 0) {
    return left;
  }
  return make_operator(Operator::ADD, {left.node, right.node});
}

Traced operator-(const Traced& left, const Traced& right) {
  double left_value, right_value;
  auto left_is_constant = is_constant(left, left_value);
  auto right_is_constant = is_constant(right, right_value);
  if (left_is_constant && right_is_constant) {
    return left_value - right_value;
  }
  if (left_is_constant && left_value == 0) {
    return -right;
  }
  if (right_is_constant && right_value == 0) {
    return left;
  }
  return make_operator(Operator::SUBTRACT, {left.node, right.node});
}

Traced operator-(const Traced& x) {
  double value;
  if (is_constant(x.node, value)) {
    return -value;
  }
  return make_operator(Operator::NEGATE, {x.node});
}

Traced operator*(const Traced& left, const Traced& right) {
  double left_value, right_value;
  auto left_is_constant = is_constant(left, left_value);
  auto right_is_constant = is_constant(right, right_value);
  if (left_is_constant && right_is_constant) {
    return left_value * right_value;
  }
  if ((left_is_constant && left_value == 0) ||
      (right_is_constant && right_value == 1)) {
    return left;
  }
  if ((right_is_constant && right_value == 0) ||
      (left_is_constant && left_value == 1)) {
    return right;
  }
  return make_operator(Operator::MULTIPLY, {left.node, right.node});
}

Traced operator/(const Traced& left, const Traced& right) {
  double left_value, right_value;
  auto left_is_constant = is_constant(left, left_value);
  auto right_is_constant = is_constant(right, right_value);
  if (left_is_constant && right_is_constant) {
    return left_value / right_value;
  }
  if ((left_is_constant && left_value == 0) ||
      (right_is_constant && right_value == 1)) {
    return left;
  }
  return make_operator(Operator::DIVIDE, {left.node, right.node});
}

Traced pow(const Traced& base, const Traced& exponent) {
  double left_value, right_value;
  auto left_is_constant = is_constant(base, left_value);
  auto right_is_constant = is_constant(exponent, right_value);
  if (left_is_constant && right_is_constant) {
    return pow(Real{left_value}, Real{right_value});
  }
  if (right_is_constant) {
    if (right_value == 0) {
      return 1;
    }
    if (right_value == 1) {
      return base;
    }
  }
  return make_operator(Operator::POW, {base.node, exponent.node});
}

Traced exp(const Traced& x) {
  double value;
  auto value_is_constant = is_constant(x, value);
  if (value_is_constant) {
    return exp(Real{value});
  }
  return make_operator(Operator::EXP, {x.node});
}

Traced log(const Traced& x) {
  double value;
  auto value_is_constant = is_constant(x, value);
  if (value_is_constant) {
    return log(Real{value});
  }
  return make_operator(Operator::LOG, {x.node});
}

Traced atan(const Traced& x) {
  double value;
  auto value_is_constant = is_constant(x, value);
  if (value_is_constant) {
    return atan(Real{value});
  }
  return make_operator(Operator::ATAN, {x.node});
}

Traced lgamma(const Traced& x) {
  double value;
  auto value_is_constant = is_constant(x, value);
  if (value_is_constant) {
    return lgamma(Real{value});
  }
  return make_operator(Operator::LGAMMA, {x.node});
}

Traced polygamma(const int n, const Traced& x) {
  double value;
  auto value_is_constant = is_constant(x, value);
  if (value_is_constant) {
    return polygamma(n, Real{value});
  }
  return make_operator(Operator::LGAMMA, {Traced{(double)n}.node, x.node});
}

Traced if_equal(
    const Traced& value,
    const Traced& comparand,
    const Traced& when_equal,
    const Traced& when_not_equal) {
  double v;
  double c;
  if (is_constant(value, v) && is_constant(comparand, c)) {
    return (v == c) ? when_equal : when_not_equal;
  }
  return make_operator(
      Operator::IF_EQUAL,
      {value.node, comparand.node, when_equal.node, when_not_equal.node});
}

Traced if_less(
    const Traced& value,
    const Traced& comparand,
    const Traced& when_less,
    const Traced& when_not_less) {
  double v;
  double c;
  if (is_constant(value, v) && is_constant(comparand, c)) {
    return (v < c) ? when_less : when_not_less;
  }
  return make_operator(
      Operator::IF_LESS,
      {value.node, comparand.node, when_less.node, when_not_less.node});
}

bool is_constant(const Traced& x, double& value) {
  if (x.op() != Operator::CONSTANT) {
    return false;
  }
  auto knode = std::dynamic_pointer_cast<const ConstantNode>(x.node);
  value = knode->value;
  return true;
}

bool is_constant(const Traced& x, const double& value) {
  double v;
  return is_constant(x, v) && v == value;
}

} // namespace beanmachine::minibmg
