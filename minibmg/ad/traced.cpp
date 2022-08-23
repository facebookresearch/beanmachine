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

Traced Traced::variable(const std::string& name, const uint sequence) {
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
Traced Traced::operator+(const Traced& other) const {
  if (this->m_op == Operator::CONSTANT && other.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*this->m_ptr).value;
    auto v2 = dynamic_cast<const TracedConstant&>(*other.m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(v1 + v2)};
  } else if (this->is_constant(0)) {
    return other;
  } else if (other.is_constant(0)) {
    return *this;
  } else {
    return Traced{Operator::ADD, make_shared<TracedOp>(TracedOp{*this, other})};
    return Traced{Operator::ADD, make_shared<TracedOp>(TracedOp{*this, other})};
  }
}

Traced Traced::operator-(const Traced& other) const {
  if (this->m_op == Operator::CONSTANT && other.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*this->m_ptr).value;
    auto v2 = dynamic_cast<const TracedConstant&>(*other.m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(v1 - v2)};
  } else if (this->is_constant(0)) {
    return -other;
  } else if (other.is_constant(0)) {
    return *this;
  } else {
    return Traced{
        Operator::SUBTRACT, make_shared<TracedOp>(TracedOp{*this, other})};
  }
}

Traced Traced::operator-() const {
  if (this->m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*this->m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(-v1)};
  } else {
    return Traced{Operator::NEGATE, make_shared<TracedOp>(TracedOp{*this})};
  }
}

Traced Traced::operator*(const Traced& other) const {
  if (this->m_op == Operator::CONSTANT && other.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*this->m_ptr).value;
    auto v2 = dynamic_cast<const TracedConstant&>(*other.m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(v1 * v2)};
  } else if (this->is_constant(0) || other.is_constant(1)) {
    return *this;
  } else if (other.is_constant(0) || this->is_constant(1)) {
    return other;
  } else {
    return Traced{
        Operator::MULTIPLY, make_shared<TracedOp>(TracedOp{*this, other})};
  }
}

Traced Traced::operator/(const Traced& other) const {
  if (this->m_op == Operator::CONSTANT && other.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*this->m_ptr).value;
    auto v2 = dynamic_cast<const TracedConstant&>(*other.m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(v1 / v2)};
  } else if (this->is_constant(0) || other.is_constant(1)) {
    return *this;
  } else {
    return Traced{
        Operator::DIVIDE, make_shared<TracedOp>(TracedOp{*this, other})};
  }
}

Traced Traced::pow(const Traced& other) const {
  if (this->m_op == Operator::CONSTANT && other.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*this->m_ptr).value;
    auto v2 = dynamic_cast<const TracedConstant&>(*other.m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(v1.pow(v2))};
  }
  double power;
  if (other.is_constant(power)) {
    if (power == 0) {
      return 1;
    }
    if (power == 1) {
      return *this;
    }
  }
  return Traced{Operator::POW, make_shared<TracedOp>(TracedOp{*this, other})};
}

Traced Traced::exp() const {
  if (this->m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*this->m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(v1.exp())};
  }
  return Traced{Operator::EXP, make_shared<TracedOp>(TracedOp{*this})};
}

Traced Traced::log() const {
  if (this->m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*this->m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(v1.log())};
  }
  return Traced{Operator::LOG, make_shared<TracedOp>(TracedOp{*this})};
}

Traced Traced::atan() const {
  if (this->m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*this->m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(v1.atan())};
  }
  return Traced{Operator::ATAN, make_shared<TracedOp>(TracedOp{*this})};
}

Traced Traced::lgamma() const {
  if (this->m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*this->m_ptr).value;
    return Traced{Operator::CONSTANT, make_shared<TracedConstant>(v1.lgamma())};
  }
  return Traced{Operator::LGAMMA, make_shared<TracedOp>(TracedOp{*this})};
}

Traced Traced::polygamma(const Traced& other) const {
  if (this->m_op == Operator::CONSTANT && other.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*this->m_ptr).value;
    auto v2 = dynamic_cast<const TracedConstant&>(*other.m_ptr).value;
    return Traced{
        Operator::CONSTANT, make_shared<TracedConstant>(v1.polygamma(v2))};
  }
  return Traced{
      Operator::POLYGAMMA, make_shared<TracedOp>(TracedOp{*this, other})};
}

Traced Traced::if_equal(
    const Traced& comparand,
    const Traced& when_equal,
    const Traced& when_not_equal) const {
  if (this->m_op == Operator::CONSTANT &&
      comparand.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*this->m_ptr).value;
    auto v2 = dynamic_cast<const TracedConstant&>(*comparand.m_ptr).value;
    return v1.if_equal(v2, when_equal, when_not_equal);
  }
  return Traced{
      Operator::IF_EQUAL,
      make_shared<TracedOp>(
          TracedOp{*this, comparand, when_equal, when_not_equal})};
}

Traced Traced::if_less(
    const Traced& comparand,
    const Traced& when_less,
    const Traced& when_not_less) const {
  if (this->m_op == Operator::CONSTANT &&
      comparand.m_op == Operator::CONSTANT) {
    auto v1 = dynamic_cast<const TracedConstant&>(*this->m_ptr).value;
    auto v2 = dynamic_cast<const TracedConstant&>(*comparand.m_ptr).value;
    return v1.if_less(v2, when_less, when_not_less);
  }
  return Traced{
      Operator::IF_LESS,
      make_shared<TracedOp>(
          TracedOp{*this, comparand, when_less, when_not_less})};
}

bool Traced::is_constant(double& value) const {
  if (this->m_op != Operator::CONSTANT) {
    return false;
  }
  auto k = dynamic_cast<const TracedConstant&>(*this->m_ptr);
  value = k.value.as_double();
  return true;
}

bool Traced::is_constant(const double& value) const {
  double v;
  return is_constant(v) && v == value;
}

} // namespace beanmachine::minibmg
