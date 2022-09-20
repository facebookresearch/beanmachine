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
#include "beanmachine/minibmg/eval.h"

namespace beanmachine::minibmg {

Traced::Traced(const Operator op, shared_ptr<const TracedBody> p)
    : m_op{op}, m_ptr{p} {}

Traced::Traced(double n)
    : m_op{Operator::CONSTANT}, m_ptr{make_shared<TracedConstant>(n)} {}

Traced Traced::variable(const std::string& name, const unsigned sequence) {
  return Traced{
      Operator::VARIABLE, make_shared<TracedVariable>(name, sequence)};
}

double Traced::as_double() const {
  if (this->m_op != Operator::CONSTANT) {
    throw EvalError(
        "constant value expected but found " +
        beanmachine::minibmg::to_string(this->m_op));
  }
  auto v1 = dynamic_cast<const TracedConstant&>(*this->m_ptr);
  return v1.value.as_double();
}

// We perform some optimizations during construction.
// It might be better to do no optimizations at this point and have a tree
// rewriter that can be reused, but for now this is a simpler approach.
Traced operator+(const Traced& left, const Traced& right) {
  if (left.m_op == Operator::CONSTANT && right.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*left.m_ptr).value;
    auto v2 = dynamic_cast<const TracedConstant&>(*right.m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(v1 + v2)};
  } else if (is_constant(left, 0)) {
    return right;
  } else if (is_constant(right, 0)) {
    return left;
  } else {
    return Traced{Operator::ADD, make_shared<TracedOp>(TracedOp{left, right})};
  }
}

Traced operator-(const Traced& left, const Traced& right) {
  if (left.m_op == Operator::CONSTANT && right.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*left.m_ptr).value;
    auto v2 = dynamic_cast<const TracedConstant&>(*right.m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(v1 - v2)};
  } else if (is_constant(left, 0)) {
    return -right;
  } else if (is_constant(right, 0)) {
    return left;
  } else {
    return Traced{
        Operator::SUBTRACT, make_shared<TracedOp>(TracedOp{left, right})};
  }
}

Traced operator-(const Traced& x) {
  if (x.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*x.m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(-v1)};
  } else {
    return Traced{Operator::NEGATE, make_shared<TracedOp>(TracedOp{x})};
  }
}

Traced operator*(const Traced& left, const Traced& right) {
  if (left.m_op == Operator::CONSTANT && right.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*left.m_ptr).value;
    auto v2 = dynamic_cast<const TracedConstant&>(*right.m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(v1 * v2)};
  } else if (is_constant(left, 0) || is_constant(right, 1)) {
    return left;
  } else if (is_constant(right, 0) || is_constant(left, 1)) {
    return right;
  } else {
    return Traced{
        Operator::MULTIPLY, make_shared<TracedOp>(TracedOp{left, right})};
  }
}

Traced operator/(const Traced& left, const Traced& right) {
  if (left.m_op == Operator::CONSTANT && right.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*left.m_ptr).value;
    auto v2 = dynamic_cast<const TracedConstant&>(*right.m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(v1 / v2)};
  } else if (is_constant(left, 0) || is_constant(right, 1)) {
    return left;
  } else {
    return Traced{
        Operator::DIVIDE, make_shared<TracedOp>(TracedOp{left, right})};
  }
}

Traced pow(const Traced& base, const Traced& exponent) {
  if (base.m_op == Operator::CONSTANT && exponent.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*base.m_ptr).value;
    auto v2 = dynamic_cast<const TracedConstant&>(*exponent.m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(pow(v1, v2))};
  }
  double power;
  if (is_constant(exponent, power)) {
    if (power == 0) {
      return 1;
    }
    if (power == 1) {
      return base;
    }
  }
  return Traced{Operator::POW, make_shared<TracedOp>(TracedOp{base, exponent})};
}

Traced exp(const Traced& x) {
  if (x.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*x.m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(exp(v1))};
  }
  return Traced{Operator::EXP, make_shared<TracedOp>(TracedOp{x})};
}

Traced log(const Traced& x) {
  if (x.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*x.m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(log(v1))};
  }
  return Traced{Operator::LOG, make_shared<TracedOp>(TracedOp{x})};
}

Traced atan(const Traced& x) {
  if (x.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*x.m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(atan(v1))};
  }
  return Traced{Operator::ATAN, make_shared<TracedOp>(TracedOp{x})};
}

Traced lgamma(const Traced& x) {
  if (x.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*x.m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(lgamma(v1))};
  }
  return Traced{Operator::LGAMMA, make_shared<TracedOp>(TracedOp{x})};
}

Traced polygamma(const int n, const Traced& x) {
  if (x.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*x.m_ptr).value;
    return Traced{
        Operator::CONSTANT, make_shared<TracedConstant>(polygamma(n, v1))};
  }
  Traced kn{Operator::CONSTANT, make_shared<TracedConstant>(n)};
  return Traced{Operator::POLYGAMMA, make_shared<TracedOp>(TracedOp{kn, x})};
}

Traced if_equal(
    const Traced& value,
    const Traced& comparand,
    const Traced& when_equal,
    const Traced& when_not_equal) {
  if (value.m_op == Operator::CONSTANT &&
      comparand.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*value.m_ptr).value;
    auto v2 = dynamic_cast<const TracedConstant&>(*comparand.m_ptr).value;
    return if_equal(v1, v2, when_equal, when_not_equal);
  }
  return Traced{
      Operator::IF_EQUAL,
      make_shared<TracedOp>(
          TracedOp{value, comparand, when_equal, when_not_equal})};
}

Traced if_less(
    const Traced& value,
    const Traced& comparand,
    const Traced& when_less,
    const Traced& when_not_less) {
  if (value.m_op == Operator::CONSTANT &&
      comparand.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*value.m_ptr).value;
    auto v2 = dynamic_cast<const TracedConstant&>(*comparand.m_ptr).value;
    return if_less(v1, v2, when_less, when_not_less);
  }
  return Traced{
      Operator::IF_LESS,
      make_shared<TracedOp>(
          TracedOp{value, comparand, when_less, when_not_less})};
}

bool is_constant(const Traced& x, double& value) {
  if (x.m_op != Operator::CONSTANT) {
    return false;
  }
  auto k = dynamic_cast<const TracedConstant&>(*x.m_ptr);
  value = k.value.as_double();
  return true;
}

bool is_constant(const Traced& x, const double& value) {
  double v;
  return is_constant(x, v) && v == value;
}

} // namespace beanmachine::minibmg
