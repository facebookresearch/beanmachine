/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include "beanmachine/minibmg/ad/number.h"
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/dedup2.h"
#include "beanmachine/minibmg/node2.h"

namespace beanmachine::minibmg {

/*
An implementation of the Number concept that simply builds an expression tree
(actually, a DAG: directed acyclic graph) representing the computation
performed.  This is also used for the fluid factory.
 */
class Traced2 {
 public:
  ScalarNode2p node;

  /* implicit */ inline Traced2(ScalarNode2p node) : node{node} {}
  /* implicit */ inline Traced2(double value)
      : node{std::make_shared<ScalarConstantNode2>(value)} {}
  /* implicit */ inline Traced2(Real value)
      : node{std::make_shared<ScalarConstantNode2>(value.as_double())} {}

  static Traced2 variable(const std::string& name, const unsigned identifier) {
    return Traced2{std::make_shared<ScalarVariableNode2>(name, identifier)};
  }

  double as_double() const;
};

Traced2 operator+(const Traced2& left, const Traced2& right);
Traced2 operator-(const Traced2& left, const Traced2& right);
Traced2 operator-(const Traced2& x);
Traced2 operator*(const Traced2& left, const Traced2& right);
Traced2 operator/(const Traced2& left, const Traced2& right);
Traced2 pow(const Traced2& base, const Traced2& exponent);
Traced2 exp(const Traced2& x);
Traced2 log(const Traced2& x);
Traced2 atan(const Traced2& x);
Traced2 lgamma(const Traced2& x);
Traced2 polygamma(int n, const Traced2& other);
Traced2 if_equal(
    const Traced2& value,
    const Traced2& comparand,
    const Traced2& when_equal,
    const Traced2& when_not_equal);
Traced2 if_less(
    const Traced2& value,
    const Traced2& comparand,
    const Traced2& when_less,
    const Traced2& when_not_less);
bool is_constant(const Traced2& x, double& value);
bool is_constant(const Traced2& x, const double& value);
std::string to_string(const Traced2& x);

static_assert(Number<Traced2>);

template <>
class DedupAdapter<Traced2> {
 public:
  std::vector<Node2p> find_roots(const Traced2& t) const {
    return {t.node};
  }
  Traced2 rewrite(
      const Traced2& t,
      const std::unordered_map<Node2p, Node2p>& map) const {
    const auto& f = map.find(t.node);
    return (f == map.end())
        ? t
        : Traced2(std::dynamic_pointer_cast<const ScalarNode2>(f->second));
  }
};

} // namespace beanmachine::minibmg

// We want to use Traced2 values in (unordered) maps and sets, so we need a good
// comparison function.  We delegate to the underlying node pointer for
// that comparison.
template <>
struct std::less<beanmachine::minibmg::Traced2> {
  bool operator()(
      const beanmachine::minibmg::Traced2& lhs,
      const beanmachine::minibmg::Traced2& rhs) const {
    static const auto x = std::less<beanmachine::minibmg::ScalarNode2p>{};
    return x(lhs.node, rhs.node);
  }
};
