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
 private:
  double value;

 public:
  INLINE double as_double() const {
    return value;
  }
  /* implicit */ inline Real(double value) : value{value} {}
  INLINE Real& operator=(const Real&) = default;
  INLINE Real operator+(Real other) const {
    return this->value + other.value;
  }
  INLINE Real operator-(Real other) const {
    return this->value - other.value;
  }
  INLINE Real operator-() const {
    return -this->value;
  }
  INLINE Real operator*(Real other) const {
    return this->value * other.value;
  }
  INLINE Real operator/(Real other) const {
    return this->value / other.value;
  }
  INLINE Real pow(Real other) const {
    return std::pow(this->value, other.value);
  }
  INLINE Real exp() const {
    return std::exp(this->value);
  }
  INLINE Real log() const {
    return std::log(this->value);
  }
  INLINE Real atan() const {
    return std::atan(this->value);
  }
  INLINE Real lgamma() const {
    return std::lgamma(this->value);
  }
  INLINE Real polygamma(Real other) const {
    // Note the order of operands here.
    return boost::math::polygamma(other.value, this->value);
  }
  template <class T>
  INLINE const T&
  if_equal(Real comparand, const T& when_equal, const T& when_not_equal) const {
    return (this->value == comparand.value) ? when_equal : when_not_equal;
  }
  template <class T>
  INLINE const T&
  if_less(Real comparand, const T& when_less, const T& when_not_less) const {
    return (this->value < comparand.value) ? when_less : when_not_less;
  }
  INLINE bool is_constant(double& value) const {
    value = this->value;
    return true;
  }
  INLINE bool is_constant(const double& value) const {
    double v = 0;
    return is_constant(v) && v == value;
  }
  inline std::string to_string() const {
    // The behavior of std::to_string(double) is that it uses a fixed number of
    // digits of precision.  We would prefer to use the minimum number of digits
    // that round-trips to the same value, so we use fmt::format.
    return fmt::format("{0}", value);
  }
};

static_assert(Number<Real>);

} // namespace beanmachine::minibmg
