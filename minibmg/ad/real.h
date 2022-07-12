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

namespace beanmachine::minibmg {

/*
 * This is intended to be a very lightweight wrapper around
 * a double that implements the Number concept.
 */
class Real {
 public:
  const double value;
  inline double as_double() const {
    return value;
  }
  /* implicit */ Real(double value) : value{value} {}
  inline Real operator+(const Real& other) const {
    return this->value + other.value;
  }
  inline Real operator-(const Real& other) const {
    return this->value - other.value;
  }
  inline Real operator-() const {
    return -this->value;
  }
  inline Real operator*(const Real& other) const {
    return this->value * other.value;
  }
  inline Real operator/(const Real& other) const {
    return this->value / other.value;
  }
  inline Real pow(const Real& other) const {
    return std::pow(this->value, other.value);
  }
  inline Real exp() const {
    return std::exp(this->value);
  }
  inline Real log() const {
    return std::log(this->value);
  }
  inline Real atan() const {
    return std::atan(this->value);
  }
  inline Real lgamma() const {
    return std::lgamma(this->value);
  }
  inline Real polygamma(const Real& other) const {
    // Note the order of operands here.
    return boost::math::polygamma(other.value, this->value);
  }
  inline Real if_equal(
      const Real& comparand,
      const Real& when_equal,
      const Real& when_not_equal) const {
    return (this->value == comparand.value) ? when_equal : when_not_equal;
  }
  inline Real if_less(
      const Real& comparand,
      const Real& when_less,
      const Real& when_not_less) const {
    return (this->value < comparand.value) ? when_less : when_not_less;
  }
  inline bool is_definitely_zero() const {
    return this->value == 0;
  }
  inline bool is_definitely_one() const {
    return this->value == 1;
  }
};

static_assert(Number<Real>);

} // namespace beanmachine::minibmg
