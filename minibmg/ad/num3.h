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
 * A simple class for holding a value and its first and second derivatives.
 * Can be used to automatically compute derivatives of more complex functions
 * by using the overloaded operators for computing new values and their
 * derivatives.  Implements the Number concept.
 */
template <class Underlying>
requires Number<Underlying>
class Num3 {
 public:
  const Underlying primal, derivative1, derivative2;
  /* implicit */ Num3(double primal);
  /* implicit */ Num3(Underlying primal);
  Num3(Underlying primal, Underlying derivative1, Underlying derivative2);
  double as_double() const;
  Num3<Underlying> operator+(const Num3<Underlying>& other) const;
  Num3<Underlying> operator-(const Num3<Underlying>& other) const;
  Num3<Underlying> operator-() const;
  Num3<Underlying> operator*(const Num3<Underlying>& other) const;
  Num3<Underlying> operator/(const Num3<Underlying>& other) const;
  Num3<Underlying> pow(const Num3<Underlying>& other) const;
  Num3<Underlying> exp() const;
  Num3<Underlying> log() const;
  Num3<Underlying> atan() const;
  bool is_definitely_zero() const;
  bool is_definitely_one() const;
};

template <class Underlying>
requires Number<Underlying> Num3<Underlying>::Num3(double primal)
    : primal{primal}, derivative1{0}, derivative2{0} {}

template <class Underlying>
requires Number<Underlying> Num3<Underlying>::Num3(Underlying primal)
    : primal{primal}, derivative1{0}, derivative2{0} {}

template <class Underlying>
requires Number<Underlying> Num3<Underlying>::Num3(
    Underlying primal,
    Underlying derivative1,
    Underlying derivative2)
    : primal{primal}, derivative1{derivative1}, derivative2{derivative2} {}

template <class Underlying>
requires Number<Underlying>
double Num3<Underlying>::as_double() const {
  return primal.as_double();
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::operator+(
    const Num3<Underlying>& other) const {
  return Num3<Underlying>(
      this->primal + other.primal,
      this->derivative1 + other.derivative1,
      this->derivative2 + other.derivative2);
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::operator-(
    const Num3<Underlying>& other) const {
  return Num3<Underlying>(
      this->primal - other.primal,
      this->derivative1 - other.derivative1,
      this->derivative2 - other.derivative2);
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::operator-()
    const {
  return Num3<Underlying>(
      -this->primal, -this->derivative1, -this->derivative2);
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::operator*(
    const Num3<Underlying>& other) const {
  // a * b
  Underlying primal = this->primal * other.primal;
  // Derivative: d/dx (a * b)
  //             = a' * b + b' * a (product rule)
  Underlying newDerivative1 =
      this->derivative1 * other.primal + other.derivative1 * this->primal;
  // Second derivative: d/dx d/dx (a * b)
  //             = d/dx (a' * b + a * b') (product rule)
  //             = d/dx (a' * b) + d/dx (a * b') // sum rule
  //             = (a'' * b + a' * b') + (a' * b' + a * b'') // product rule
  //             = a'' * b + 2 * a' * b' + a * b'' // gather terms
  Underlying newDerivative2 = this->derivative2 * other.primal +
      2 * this->derivative1 * other.derivative1 +
      this->primal * other.derivative2;
  return Num3<Underlying>{primal, newDerivative1, newDerivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::operator/(
    const Num3<Underlying>& other) const {
  // TODO: simplify this
  Underlying t0 = other.primal * other.primal;
  Underlying t1 = 1.0f / t0;
  Underlying t2 =
      (other.primal * this->derivative1) - (this->primal * other.derivative1);
  Underlying t3 = other.primal * other.derivative1;

  Underlying newPrimal = this->primal / other.primal;
  Underlying newDerivative1 = t1 * t2;
  Underlying newDerivative2 = (t1 *
                               (((other.primal * this->derivative2) +
                                 (this->derivative1 * other.derivative1)) -
                                ((this->primal * other.derivative2) +
                                 (other.derivative1 * this->derivative1)))) -
      (t2 * ((1.0f / (t0 * t0)) * (t3 + t3)));
  return Num3<Underlying>{newPrimal, newDerivative1, newDerivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::pow(
    const Num3<Underlying>& other)
const {
  // TODO: simplify this.
  const Underlying t0 = this->primal.log();
  const Underlying newPrimal = (other.primal * t0).exp();
  const Underlying t2 = this->derivative1 / this->primal;
  const Underlying t3 = (other.primal * t2) + (t0 * other.derivative1);
  const Underlying newDerivative1 = newPrimal * t3;

  const Underlying newDerivative2 =
      (newPrimal *
       (((other.primal *
          ((this->derivative2 / this->primal) -
           (this->derivative1 *
            (this->derivative1 * this->primal.pow(-2.0f))))) +
         (t2 * other.derivative1)) +
        ((t0 * other.derivative2) + (other.derivative1 * t2)))) +
      (t3 * newDerivative1);
  return Num3{newPrimal, newDerivative1, newDerivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::exp()
const {
  const Underlying t6 = this->primal.exp();
  const Underlying t7 = t6 * this->derivative1;

  const Underlying newPrimal = t6;
  const Underlying newDerivative1 = t7;
  const Underlying newDerivative2 =
      (t6 * this->derivative2) + (this->derivative1 * t7);
  return Num3{newPrimal, newDerivative1, newDerivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::log()
const {
  const Underlying newPrimal = this->primal.log();
  const Underlying newDerivative1 = this->derivative1 / this->primal;
  const Underlying newDerivative2 = (this->derivative2 / this->primal) -
      (this->derivative1 * (this->derivative1 * this->primal.pow(-2.0f)));
  return Num3{newPrimal, newDerivative1, newDerivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::atan()
const {
  // TODO: simplify this
  const Underlying t0 = this->primal.pow(2.0f) + 1.0f;
  const Underlying t1 = 1.0f / t0;

  const Underlying newPrimal = this->primal.atan();
  const Underlying newDerivative1 = t1 * this->derivative1;
  const Underlying newDerivative2 = (t1 * this->derivative2) -
      (this->derivative1 *
       ((1.0f / (t0 * t0)) * (this->derivative1 * (2.0f * this->primal))));
  return Num3{newPrimal, newDerivative1, newDerivative2};
}

template <class Underlying>
requires Number<Underlying>
bool Num3<Underlying>::is_definitely_zero() const {
  return this->primal.is_definitely_zero() &&
      this->derivative1.is_definitely_zero() &&
      this->derivative2.is_definitely_zero();
}

template <class Underlying>
requires Number<Underlying>
bool Num3<Underlying>::is_definitely_one() const {
  return this->primal.is_definitely_one() &&
      this->derivative1.is_definitely_zero() &&
      this->derivative2.is_definitely_zero();
}

// Check that Num3<Real> satisfies Number
static_assert(Number<Num3<Real>>);

} // namespace beanmachine::minibmg
