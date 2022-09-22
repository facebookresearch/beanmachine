/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <stdexcept>
#include <vector>
#include "beanmachine/minibmg/ad/number.h"
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/node.h"

namespace beanmachine::minibmg {

/*
An implementation of the Number concept that simply builds an expression tree
(actually, a DAG: directed acyclic graph) representing the computation
performed.  This is also used for the fluent factory.
 */
class Traced {
 public:
  Nodep node;

  inline Traced(Nodep node) : node{node} {
    if (node->type != Type::REAL) {
      throw std::invalid_argument("node is not a value");
    }
  }
  /* implicit */ inline Traced(double value)
      : node{std::make_shared<ConstantNode>(value)} {}
  /* implicit */ inline Traced(Real value)
      : node{std::make_shared<ConstantNode>(value.as_double())} {}
  inline Operator op() const {
    return node->op;
  }

  static Traced variable(const std::string& name, const unsigned identifier) {
    return Traced{std::make_shared<VariableNode>(name, identifier)};
  }

  double as_double() const;
};

Traced operator+(const Traced& left, const Traced& right);
Traced operator-(const Traced& left, const Traced& right);
Traced operator-(const Traced& x);
Traced operator*(const Traced& left, const Traced& right);
Traced operator/(const Traced& left, const Traced& right);
Traced pow(const Traced& base, const Traced& exponent);
Traced exp(const Traced& x);
Traced log(const Traced& x);
Traced atan(const Traced& x);
Traced lgamma(const Traced& x);
Traced polygamma(int n, const Traced& other);
Traced if_equal(
    const Traced& value,
    const Traced& comparand,
    const Traced& when_equal,
    const Traced& when_not_equal);
Traced if_less(
    const Traced& value,
    const Traced& comparand,
    const Traced& when_less,
    const Traced& when_not_less);
bool is_constant(const Traced& x, double& value);
bool is_constant(const Traced& x, const double& value);
std::string to_string(const Traced& x);

static_assert(Number<Traced>);

} // namespace beanmachine::minibmg

// We want to use Traced values in (unordered) maps and sets, so we need a good
// comparison function.  We delegate to the underlying node pointer for
// that comparison.
template <>
struct ::std::less<beanmachine::minibmg::Traced> {
  bool operator()(
      const beanmachine::minibmg::Traced& lhs,
      const beanmachine::minibmg::Traced& rhs) const {
    static const auto x = ::std::less<beanmachine::minibmg::Nodep>{};
    return x(lhs.node, rhs.node);
  }
};
