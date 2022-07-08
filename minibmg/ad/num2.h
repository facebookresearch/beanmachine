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
 public:
  const Underlying primal, derivative1;
  /* implicit */ Num2(double primal);
  /* implicit */ Num2(Underlying primal);
  Num2(Underlying primal, Underlying derivative1);
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
  bool is_definitely_zero() const;
  bool is_definitely_one() const;
};

template <class Underlying>
requires Number<Underlying> Num2<Underlying>::Num2(double primal)
    : primal(primal), derivative1(0.0) {}

template <class Underlying>
requires Number<Underlying> Num2<Underlying>::Num2(Underlying primal)
    : primal(primal), derivative1(0.0) {}

template <class Underlying>
requires Number<Underlying> Num2<Underlying>::Num2(
    Underlying primal,
    Underlying derivative1)
    : primal(primal), derivative1(derivative1) {}

template <class Underlying>
requires Number<Underlying>
double Num2<Underlying>::as_double() const {
  return primal.as_double();
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::operator+(
    const Num2<Underlying>& other) const {
  return Num2<Underlying>{
      this->primal + other.primal, this->derivative1 + other.derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::operator-(
    const Num2<Underlying>& other) const {
  return Num2<Underlying>{
      this->primal - other.primal, this->derivative1 - other.derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::operator-()
    const {
  return Num2<Underlying>{-this->primal, -this->derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::operator*(
    const Num2<Underlying>& other) const {
  return Num2<Underlying>{
      this->primal * other.primal,
      this->primal * other.derivative1 + this->derivative1 * other.primal};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::operator/(
    const Num2<Underlying>& other) const {
  Underlying t0 = other.primal * other.primal;
  Underlying newPrimal = this->primal / other.primal;
  Underlying newDerivative1 = ((other.primal * this->derivative1) -
                               (this->primal * other.derivative1)) /
      t0;
  return Num2<Underlying>{newPrimal, newDerivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::pow(
    const Num2<Underlying>& other)
const {
  Underlying newPrimal = this->primal.pow(other.primal);
  Underlying newDerivative1 = newPrimal *
      ((other.primal * this->derivative1 / this->primal) +
       (this->primal.log() * other.derivative1));
  return Num2<Underlying>{newPrimal, newDerivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::exp()
const {
  Underlying t6 = this->primal.exp();
  Underlying newPrimal = t6;
  Underlying newDerivative1 = t6 * this->derivative1;
  return Num2{newPrimal, newDerivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::log()
const {
  Underlying newPrimal = this->primal.log();
  Underlying newDerivative1 = this->derivative1 / this->primal;
  return Num2{newPrimal, newDerivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> Num2<Underlying>::atan()
const {
  Underlying newPrimal = this->primal.atan();
  Underlying newDerivative1 =
      this->derivative1 / (this->primal * this->primal + 1.0f);
  return Num2{newPrimal, newDerivative1};
}

template <class Underlying>
requires Number<Underlying>
bool Num2<Underlying>::is_definitely_zero() const {
  return this->primal.is_definitely_zero() &&
      this->derivative1.is_definitely_zero();
}

template <class Underlying>
requires Number<Underlying>
bool Num2<Underlying>::is_definitely_one() const {
  return this->primal.is_definitely_one() &&
      this->derivative1.is_definitely_zero();
}

// Check that Num2<Real> satisfies Number
static_assert(Number<Num2<Real>>);

} // namespace beanmachine::minibmg
