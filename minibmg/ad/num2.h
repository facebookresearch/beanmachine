/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cassert>
#include <cmath>
#include <cstdlib>
#include "beanmachine/minibmg/ad/number.h"
#include "beanmachine/minibmg/ad/real.h"

namespace beanmachine::minibmg {

/*
 * A simple class for holding a value and its first derivative.
 * Can be used to automatically compute derivatives of more complex functions
 * by using the overloaded operators for computing new values and their
 * derivatives.  Implements the Number concept.
 */
template <class Underlying>
requires Number<Underlying>
class Num2 {
 private:
  Underlying m_primal, m_derivative1;

 public:
  /* implicit */ Num2(double primal);
  /* implicit */ Num2(Underlying primal);
  Num2(Underlying primal, Underlying derivative1);
  Num2(const Num2<Underlying>& other);
  Num2<Underlying>& operator=(const Num2<Underlying>& other);
  Underlying primal() const;
  Underlying derivative1() const;
  double as_double() const;
  Num2<Underlying> operator+(const Num2<Underlying>& other) const;
  Num2<Underlying> operator-(const Num2<Underlying>& other) const;
  Num2<Underlying> operator-() const;
  Num2<Underlying> operator*(const Num2<Underlying>& other) const;
  Num2<Underlying> operator/(const Num2<Underlying>& other) const;
  Num2<Underlying> pow(const Num2<Underlying>& other) const;
  Num2<Underlying> exp() const;
  Num2<Underlying> log() const;
  Num2<Underlying> atan() const;
  Num2<Underlying> lgamma() const;
  Num2<Underlying> polygamma(double n) const;
  Num2<Underlying> if_equal(
      const Num2<Underlying>& comparand,
      const Num2<Underlying>& when_equal,
      const Num2<Underlying>& when_not_equal) const;
  Num2<Underlying> if_less(
      const Num2<Underlying>& comparand,
      const Num2<Underlying>& when_less,
      const Num2<Underlying>& when_not_less) const;
  bool is_constant(double& value) const;
  bool is_constant(const double& value) const;
};

template <class Underlying>
requires Number<Underlying> Num2<Underlying>::Num2(double primal)
    : m_primal(primal), m_derivative1(0.0) {}

template <class Underlying>
requires Number<Underlying> Num2<Underlying>::Num2(Underlying primal)
    : m_primal(primal), m_derivative1(0.0) {}

template <class Underlying>
requires Number<Underlying> Num2<Underlying>::Num2(
    Underlying primal,
    Underlying derivative1)
    : m_primal(primal), m_derivative1(derivative1) {}

template <class Underlying>
requires Number<Underlying> Num2<Underlying>::Num2(
    const Num2<Underlying>& other)
    : m_primal(other.m_primal), m_derivative1(other.m_derivative1) {}

template <class Underlying>
requires Number<Underlying> Num2<Underlying>
&Num2<Underlying>::operator=(const Num2<Underlying>& other) {
  this->m_primal = other.m_primal;
  this->m_derivative1 = other.m_derivative1;
  return *this;
}

template <class Underlying>
requires Number<Underlying> Underlying Num2<Underlying>::primal()
const {
  return m_primal;
}

template <class Underlying>
requires Number<Underlying> Underlying Num2<Underlying>::derivative1()
const {
  return m_derivative1;
}

template <class Underlying>
requires Number<Underlying>
double Num2<Underlying>::as_double() const {
  return m_primal.as_double();
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::operator+(
    const Num2<Underlying>& other) const {
  return Num2<Underlying>{
      this->m_primal + other.m_primal,
      this->m_derivative1 + other.m_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::operator-(
    const Num2<Underlying>& other) const {
  return Num2<Underlying>{
      this->m_primal - other.m_primal,
      this->m_derivative1 - other.m_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::operator-()
    const {
  return Num2<Underlying>{-this->m_primal, -this->m_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::operator*(
    const Num2<Underlying>& other) const {
  return Num2<Underlying>{
      this->m_primal * other.m_primal,
      this->m_primal * other.m_derivative1 +
          this->m_derivative1 * other.m_primal};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::operator/(
    const Num2<Underlying>& other) const {
  // a / b
  Underlying new_primal = this->m_primal / other.m_primal;

  // Using https://www.wolframalpha.com/
  // D[a(x) / b(x), x] -> (b[x] a'[x] - a[x] b'[x])/b[x]^2
  auto other2 = other.m_primal * other.m_primal;
  Underlying new_derivative1 = (other.m_primal * this->m_derivative1 -
                                this->m_primal * other.m_derivative1) /
      other2;
  return Num2<Underlying>{new_primal, new_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::pow(
    const Num2<Underlying>& other)
const {
  double power;
  // shortcut some cases.
  if (other.is_constant(power)) {
    if (power == 0)
      return 1;
    if (power == 1)
      return *this;
    if (power == 2)
      return (*this) * (*this);
    if (power == -1)
      return 1 / (*this);
  }

  const Underlying new_primal = this->m_primal.pow(other.m_primal);

  // From https://www.wolframalpha.com/
  // D[a(x) ^ (b(x)), x] ->
  //          a[x]^(-1 + b[x]) (b[x] a'[x] + a[x] Log[a[x]] b'[x])
  //        = a[x]^b[x] (b[x] a'[x] / a[x] + Log[a[x]] b'[x])

  // We avoid using the log (e.g. of a negative number) when not needed.
  // t0 = Log[a[x]] b'[x]
  const Underlying t0 = other.m_derivative1.is_constant(0)
      ? 0
      : this->m_primal.log() * other.m_derivative1;

  // b[x] * a'[x] / a[x] + Log[a[x]] b'[x]
  const Underlying x0 =
      other.m_primal * this->m_derivative1 / this->m_primal + t0;

  // a[x]^b[x] (b[x] a'[x] / a[x] + Log[a[x]] b'[x])
  Underlying new_derivative1 = new_primal * x0;

  return Num2<Underlying>{new_primal, new_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::exp()
const {
  Underlying new_primal = this->m_primal.exp();
  Underlying new_derivative1 = new_primal * this->m_derivative1;
  return Num2<Underlying>{new_primal, new_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::log()
const {
  Underlying new_primal = this->m_primal.log();
  Underlying new_derivative1 = this->m_derivative1 / this->m_primal;
  return Num2<Underlying>{new_primal, new_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::atan()
const {
  Underlying new_primal = this->m_primal.atan();
  Underlying new_derivative1 =
      this->m_derivative1 / (this->m_primal * this->m_primal + 1.0f);
  return Num2<Underlying>{new_primal, new_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::lgamma()
const {
  Underlying new_primal = this->m_primal.lgamma();
  Underlying new_derivative1 =
      this->m_derivative1 * this->m_primal.polygamma(0);
  return Num2<Underlying>{new_primal, new_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::polygamma(
    double n)
const {
  Underlying new_primal = this->m_primal.polygamma(n);
  Underlying new_derivative1 =
      this->m_derivative1 * this->m_primal.polygamma(n + 1);
  return Num2<Underlying>{new_primal, new_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::if_equal(
    const Num2<Underlying>& comparand,
    const Num2<Underlying>& when_equal,
    const Num2<Underlying>& when_not_equal)
const {
  // Note: we discard and ignore this->m_derivative1 and
  // comparand->m_derivative1
  Underlying new_primal = this->m_primal.if_equal(
      comparand.m_primal, when_equal.m_primal, when_not_equal.m_primal);
  Underlying new_derivative1 = this->m_primal.if_equal(
      comparand.m_primal,
      when_equal.m_derivative1,
      when_not_equal.m_derivative1);
  return Num2<Underlying>{new_primal, new_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::if_less(
    const Num2<Underlying>& comparand,
    const Num2<Underlying>& when_less,
    const Num2<Underlying>& when_not_less)
const {
  // Note: we discard and ignore this->m_derivative1 and
  // comparand->m_derivative1
  Underlying new_primal = this->m_primal.if_less(
      comparand.m_primal, when_less.m_primal, when_not_less.m_primal);
  Underlying new_derivative1 = this->m_primal.if_less(
      comparand.m_primal, when_less.m_derivative1, when_not_less.m_derivative1);
  return Num2<Underlying>{new_primal, new_derivative1};
}

template <class Underlying>
requires Number<Underlying>
bool Num2<Underlying>::is_constant(double& value) const {
  return this->m_derivative1.is_constant(0) &&
      this->m_primal.is_constant(value);
}

template <class Underlying>
requires Number<Underlying>
bool Num2<Underlying>::is_constant(const double& value) const {
  return this->m_derivative1.is_constant(0) &&
      this->m_primal.is_constant(value);
}

static_assert(Number<Num2<Real>>);

} // namespace beanmachine::minibmg
