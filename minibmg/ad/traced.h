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
#include "beanmachine/minibmg/dedup.h"
#include "beanmachine/minibmg/localopt.h"
#include "beanmachine/minibmg/node.h"

namespace beanmachine::minibmg {

/*
An implementation of the Number concept that simply builds an expression tree
(actually, a DAG: directed acyclic graph) representing the computation
performed.  This is also used for the fluid factory.
 */
class Traced {
 public:
  ScalarNodep node;

  /* implicit */ inline Traced(ScalarNodep node) : node{node} {}
  /* implicit */ inline Traced(double value)
      : node{std::make_shared<ScalarConstantNode>(value)} {}
  /* implicit */ inline Traced(Real value)
      : node{std::make_shared<ScalarConstantNode>(value.as_double())} {}

  static Traced variable(const std::string& name, const unsigned identifier) {
    return Traced{std::make_shared<ScalarVariableNode>(name, identifier)};
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
Traced polygamma(int n, const Traced& x);
Traced log1p(const Traced& x);
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

template <>
class DedupAdapter<Traced> {
 public:
  std::vector<Nodep> find_roots(const Traced& t) const {
    return {t.node};
  }
  Traced rewrite(const Traced& t, const std::unordered_map<Nodep, Nodep>& map)
      const {
    const auto& f = map.find(t.node);
    return (f == map.end())
        ? t
        : Traced(std::dynamic_pointer_cast<const ScalarNode>(f->second));
  }
};

} // namespace beanmachine::minibmg

// We want to use Traced values in (unordered) maps and sets, so we need a good
// comparison function.  We delegate to the underlying node pointer for
// that comparison.
template <>
struct std::less<beanmachine::minibmg::Traced> {
  bool operator()(
      const beanmachine::minibmg::Traced& lhs,
      const beanmachine::minibmg::Traced& rhs) const {
    static const auto x = std::less<beanmachine::minibmg::ScalarNodep>{};
    return x(lhs.node, rhs.node);
  }
};
