/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <boost/math/special_functions/polygamma.hpp>
#include <fmt/format.h>
#include <cmath>
#include "beanmachine/minibmg/ad/number.h"

// Set this to the keyword "inline" in production use.
// Set it to nothing to make debugging simpler.
#define INLINE inline

namespace beanmachine::minibmg {

/*
 * This is intended to be a very lightweight wrapper around
 * a double that implements the Number concept.
 */
class Real {
 public:
  double value;

  INLINE double as_double() const {
    return value;
  }
  /* implicit */ inline Real(double value) : value{value} {}
  INLINE Real& operator=(const Real&) = default;
};

INLINE Real operator+(const Real left, const Real right) {
  return left.value + right.value;
}
INLINE Real operator-(const Real left, const Real right) {
  return left.value - right.value;
}
INLINE Real operator-(const Real x) {
  return -x.value;
}
INLINE Real operator*(const Real left, const Real right) {
  return left.value * right.value;
}
INLINE Real operator/(const Real left, const Real right) {
  return left.value / right.value;
}
INLINE Real pow(const Real left, const Real right) {
  return std::pow(left.value, right.value);
}
INLINE Real exp(const Real x) {
  return std::exp(x.value);
}
INLINE Real log(const Real x) {
  return std::log(x.value);
}
INLINE Real atan(const Real x) {
  return std::atan(x.value);
}
INLINE Real lgamma(const Real x) {
  return std::lgamma(x.value);
}
INLINE Real polygamma(const int n, const Real x) {
  return boost::math::polygamma(n, x.value);
}
template <class T>
INLINE const T& if_equal(
    const Real value,
    const Real comparand,
    const T& when_equal,
    const T& when_not_equal) {
  return (value.value == comparand.value) ? when_equal : when_not_equal;
}
template <class T>
INLINE const T& if_less(
    const Real value,
    const Real comparand,
    const T& when_less,
    const T& when_not_less) {
  return (value.value < comparand.value) ? when_less : when_not_less;
}
INLINE bool is_constant(const Real x, double& value) {
  value = x.value;
  return true;
}
INLINE bool is_constant(const Real x, const double& value) {
  double v = 0;
  return is_constant(x, v) && v == value;
}
inline std::string to_string(const Real x) {
  // The behavior of std::to_string(double) is that it uses a fixed number of
  // digits of precision.  We would prefer to use the minimum number of digits
  // that round-trips to the same value, so we use fmt::format.
  return fmt::format("{0}", x.value);
}

static_assert(Number<const Real>);

} // namespace beanmachine::minibmg
