/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

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
  inline bool is_definitely_zero() const {
    return this->value == 0;
  }
  inline bool is_definitely_one() const {
    return this->value == 1;
  }
};

static_assert(Number<Real>);

} // namespace beanmachine::minibmg
