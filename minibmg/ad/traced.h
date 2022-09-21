/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <boost/math/special_functions/polygamma.hpp>
#include <cmath>
#include <memory>
#include <vector>
#include "beanmachine/minibmg/ad/number.h"
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/minibmg.h"

namespace beanmachine::minibmg {

using namespace std;

class TracedBody;

/*
An implementation of the Number concept that simply builds an expression tree
(actually, a DAG: directed acyclic graph) representing the computation
performed.
 */
class Traced {
 public:
  Operator m_op;
  shared_ptr<const TracedBody> m_ptr;

  /* implicit */ Traced(double d);
  static Traced variable(const std::string& name, const unsigned identifier);
  inline Operator op() const {
    return m_op;
  }
  inline shared_ptr<const TracedBody> ptr() const {
    return m_ptr;
  }

  double as_double() const;

  Traced(const Operator op, shared_ptr<const TracedBody> p);
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

class TracedBody {
 public:
  virtual ~TracedBody() {}
};

class TracedVariable : public TracedBody {
 public:
  const string name;
  const unsigned sequence;
  TracedVariable(const std::string& name, const unsigned sequence)
      : name{name}, sequence{sequence} {}
};
class TracedConstant : public TracedBody {
 public:
  const Real value;
  /* implicit */ TracedConstant(const Real value) : value{value} {}
};
class TracedOp : public TracedBody {
 public:
  const std::vector<Traced> args;
  explicit TracedOp(std::initializer_list<Traced> args) : args{args} {}
};

static_assert(Number<Traced>);

} // namespace beanmachine::minibmg

// We want to use Traced values in (unordered) maps and sets, so we need a good
// comparison function.  We delegate to the underlying TracedBody pointer for
// that comparison.
template <>
struct ::std::less<beanmachine::minibmg::Traced> {
  bool operator()(
      const beanmachine::minibmg::Traced& lhs,
      const beanmachine::minibmg::Traced& rhs) const {
    static const auto x =
        ::std::less<const beanmachine::minibmg::TracedBody*>{};
    return x(lhs.ptr().get(), rhs.ptr().get());
  }
};
