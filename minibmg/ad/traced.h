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
 private:
  Operator m_op;
  shared_ptr<const TracedBody> m_ptr;

 public:
  /* implicit */ Traced(double d);
  static Traced variable(const std::string& name, const uint sequence);
  inline Operator op() const {
    return m_op;
  }
  inline shared_ptr<const TracedBody> ptr() const {
    return m_ptr;
  }

  double as_double() const;
  Traced operator+(const Traced& other) const;
  Traced operator-(const Traced& other) const;
  Traced operator-() const;
  Traced operator*(const Traced& other) const;
  Traced operator/(const Traced& other) const;
  Traced pow(const Traced& other) const;
  Traced exp() const;
  Traced log() const;
  Traced atan() const;
  Traced lgamma() const;
  Traced polygamma(const Traced& other) const;
  Traced if_equal(
      const Traced& comparand,
      const Traced& when_equal,
      const Traced& when_not_equal) const;
  Traced if_less(
      const Traced& comparand,
      const Traced& when_less,
      const Traced& when_not_less) const;
  bool is_constant(double& value) const;
  bool is_constant(const double& value) const;
  std::string to_string() const;

 protected:
  Traced(const Operator op, shared_ptr<const TracedBody> p);
};

class TracedBody {
 public:
  virtual ~TracedBody() {}
};

class TracedVariable : public TracedBody {
 public:
  const string name;
  const uint sequence;
  TracedVariable(const std::string& name, const uint sequence)
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
