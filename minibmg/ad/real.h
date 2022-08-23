/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <boost/math/special_functions/polygamma.hpp>
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
  INLINE Real
  if_equal(Real comparand, Real when_equal, Real when_not_equal) const {
    return (this->value == comparand.value) ? when_equal : when_not_equal;
  }
  INLINE Real
  if_less(Real comparand, Real when_less, Real when_not_less) const {
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
};

static_assert(Number<Real>);

} // namespace beanmachine::minibmg
