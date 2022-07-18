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
 private:
  Underlying m_primal, m_derivative1, m_derivative2;

 public:
  /* implicit */ Num3(double primal);
  /* implicit */ Num3(Underlying primal);
  Num3(Underlying primal, Underlying derivative1, Underlying derivative2);
  Num3(const Num3<Underlying>& other);
  Num3<Underlying>& operator=(const Num3<Underlying>& other);
  Underlying primal() const;
  Underlying derivative1() const;
  Underlying derivative2() const;
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
  Num3<Underlying> lgamma() const;
  Num3<Underlying> polygamma(double n) const;
  Num3<Underlying> if_equal(
      const Num3<Underlying>& comparand,
      const Num3<Underlying>& when_equal,
      const Num3<Underlying>& when_not_equal) const;
  Num3<Underlying> if_less(
      const Num3<Underlying>& comparand,
      const Num3<Underlying>& when_less,
      const Num3<Underlying>& when_not_less) const;
  bool is_definitely_zero() const;
  bool is_definitely_one() const;
};

template <class Underlying>
requires Number<Underlying> Num3<Underlying>::Num3(double primal)
    : m_primal{primal}, m_derivative1{0}, m_derivative2{0} {}

template <class Underlying>
requires Number<Underlying> Num3<Underlying>::Num3(Underlying primal)
    : m_primal{primal}, m_derivative1{0}, m_derivative2{0} {}

template <class Underlying>
requires Number<Underlying> Num3<Underlying>::Num3(
    Underlying primal,
    Underlying derivative1,
    Underlying derivative2)
    : m_primal{primal},
      m_derivative1{derivative1},
      m_derivative2{derivative2} {}

template <class Underlying>
requires Number<Underlying> Num3<Underlying>::Num3(
    const Num3<Underlying>& other)
    : m_primal(other.m_primal),
      m_derivative1(other.m_derivative1),
      m_derivative2(other.m_derivative2) {}

template <class Underlying>
requires Number<Underlying> Num3<Underlying>
&Num3<Underlying>::operator=(const Num3<Underlying>& other) {
  this->m_primal = other.m_primal;
  this->m_derivative1 = other.m_derivative1;
  this->m_derivative2 = other.m_derivative2;
  return *this;
}

template <class Underlying>
requires Number<Underlying> Underlying Num3<Underlying>::primal()
const {
  return m_primal;
}

template <class Underlying>
requires Number<Underlying> Underlying Num3<Underlying>::derivative1()
const {
  return m_derivative1;
}

template <class Underlying>
requires Number<Underlying> Underlying Num3<Underlying>::derivative2()
const {
  return m_derivative2;
}

template <class Underlying>
requires Number<Underlying>
double Num3<Underlying>::as_double() const {
  return m_primal.as_double();
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::operator+(
    const Num3<Underlying>& other) const {
  return Num3<Underlying>{
      this->m_primal + other.m_primal,
      this->m_derivative1 + other.m_derivative1,
      this->m_derivative2 + other.m_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::operator-(
    const Num3<Underlying>& other) const {
  return Num3<Underlying>{
      this->m_primal - other.m_primal,
      this->m_derivative1 - other.m_derivative1,
      this->m_derivative2 - other.m_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::operator-()
    const {
  return Num3<Underlying>{
      -this->m_primal, -this->m_derivative1, -this->m_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::operator*(
    const Num3<Underlying>& other) const {
  // a * b
  Underlying new_primal = this->m_primal * other.m_primal;
  // Derivative: d/dx (a * b)
  //             = a' * b + b' * a (product rule)
  Underlying new_derivative1 = this->m_derivative1 * other.m_primal +
      other.m_derivative1 * this->m_primal;
  // Second derivative: d/dx d/dx (a * b)
  //             = d/dx (a' * b + a * b') (product rule)
  //             = d/dx (a' * b) + d/dx (a * b') // sum rule
  //             = (a'' * b + a' * b') + (a' * b' + a * b'') // product rule
  //             = a'' * b + 2 * a' * b' + a * b'' // gather terms
  Underlying new_derivative2 = this->m_derivative2 * other.m_primal +
      2 * this->m_derivative1 * other.m_derivative1 +
      this->m_primal * other.m_derivative2;
  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::operator/(
    const Num3<Underlying>& other) const {
  // TODO: simplify this
  Underlying t0 = other.m_primal * other.m_primal;
  Underlying t1 = 1.0f / t0;
  Underlying t2 = (other.m_primal * this->m_derivative1) -
      (this->m_primal * other.m_derivative1);
  Underlying t3 = other.m_primal * other.m_derivative1;

  Underlying new_primal = this->m_primal / other.m_primal;
  Underlying new_derivative1 = t1 * t2;
  Underlying new_derivative2 =
      (t1 *
       (((other.m_primal * this->m_derivative2) +
         (this->m_derivative1 * other.m_derivative1)) -
        ((this->m_primal * other.m_derivative2) +
         (other.m_derivative1 * this->m_derivative1)))) -
      (t2 * ((1.0f / (t0 * t0)) * (t3 + t3)));
  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::pow(
    const Num3<Underlying>& other)
const {
  if (other.m_derivative1.is_definitely_zero() &&
      other.m_derivative2.is_definitely_zero()) {
    const Underlying new_primal = this->m_primal.pow(other.m_primal);
    const Underlying new_derivative1 =
        new_primal * other.m_primal * this->m_derivative1 / this->m_primal;
    const Underlying new_derivative2 = new_derivative1 * (other.m_primal - 1) *
        this->m_derivative1 / this->m_primal;
    return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
  } else {
    // TODO: simplify this.
    const Underlying t0 = this->m_primal.log();
    const Underlying new_primal = (other.m_primal * t0).exp();
    const Underlying t2 = this->m_derivative1 / this->m_primal;
    const Underlying t3 = (other.m_primal * t2) + (t0 * other.m_derivative1);
    const Underlying new_derivative1 = new_primal * t3;

    const Underlying new_derivative2 =
        (new_primal *
         (((other.m_primal *
            ((this->m_derivative2 / this->m_primal) -
             (this->m_derivative1 *
              (this->m_derivative1 * this->m_primal.pow(-2.0f))))) +
           (t2 * other.m_derivative1)) +
          ((t0 * other.m_derivative2) + (other.m_derivative1 * t2)))) +
        (t3 * new_derivative1);
    return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
  }
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::exp()
const {
  const Underlying t6 = this->m_primal.exp();
  const Underlying t7 = t6 * this->m_derivative1;

  const Underlying new_primal = t6;
  const Underlying new_derivative1 = t7;
  const Underlying new_derivative2 =
      (t6 * this->m_derivative2) + (this->m_derivative1 * t7);
  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::log()
const {
  const Underlying new_primal = this->m_primal.log();
  const Underlying new_derivative1 = this->m_derivative1 / this->m_primal;
  const Underlying new_derivative2 = (this->m_derivative2 / this->m_primal) -
      (this->m_derivative1 * (this->m_derivative1 * this->m_primal.pow(-2.0f)));
  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::atan()
const {
  // TODO: simplify this
  const Underlying t0 = this->m_primal.pow(2.0f) + 1.0f;
  const Underlying t1 = 1.0f / t0;

  const Underlying new_primal = this->m_primal.atan();
  const Underlying new_derivative1 = t1 * this->m_derivative1;
  const Underlying new_derivative2 = (t1 * this->m_derivative2) -
      (this->m_derivative1 *
       ((1.0f / (t0 * t0)) * (this->m_derivative1 * (2.0f * this->m_primal))));
  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::lgamma()
const {
  // Note: First order chain rule:
  // d/dx[f(g(x))]
  //     = f’(g(x)) g’(x)
  // Second order chain rule:
  // d^2/dx^2[f(g(x))]
  //     = d/dx[f’(g(x)) g’(x)] // first order chain rule
  //     = d/dx[f’(g(x))] g’(x) // product rule
  //       + d/dx[g’(x)] f’(g(x))
  //     = f’’(g(x)) g’(x) g’(x) // first order chain rule
  //       + g’’(x) f’(g(x))
  //     = f’’(g(x)) g’(x) g’(x) + g’’(x) f’(g(x))
  // Here f is lgamma, g is the value of the parameter to the function
  // being differentiated, and g' and g'' are the incoming gradients of
  // the value.
  Underlying new_primal = this->m_primal.lgamma();
  auto t1 = this->m_primal.polygamma(0); // f’(g(x))
  Underlying new_derivative1 = this->m_derivative1 * t1;
  Underlying new_derivative2 =
      this->m_primal.polygamma(1) * this->m_derivative1 * this->m_derivative1 +
      this->m_derivative2 * t1;
  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::polygamma(
    double n)
const {
  Underlying new_primal = this->m_primal.polygamma(n);
  auto t1 = this->m_primal.polygamma(n + 1);
  Underlying new_derivative1 = this->m_derivative1 * t1;
  Underlying new_derivative2 = this->m_primal.polygamma(n + 2) *
          this->m_derivative1 * this->m_derivative1 +
      this->m_derivative2 * t1;
  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::if_equal(
    const Num3<Underlying>& comparand,
    const Num3<Underlying>& when_equal,
    const Num3<Underlying>& when_not_equal)
const {
  // Note: we discard and ignore this->derivative* and comparand->derivative*
  Underlying new_primal = this->m_primal.if_equal(
      comparand.m_primal, when_equal.m_primal, when_not_equal.m_primal);
  Underlying new_derivative1 = this->m_primal.if_equal(
      comparand.m_primal,
      when_equal.m_derivative1,
      when_not_equal.m_derivative1);
  Underlying new_derivative2 = this->m_primal.if_equal(
      comparand.m_primal,
      when_equal.m_derivative2,
      when_not_equal.m_derivative2);
  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::if_less(
    const Num3<Underlying>& comparand,
    const Num3<Underlying>& when_less,
    const Num3<Underlying>& when_not_less)
const {
  // Note: we discard and ignore this->derivative* and comparand->derivative*
  Underlying new_primal = this->m_primal.if_less(
      comparand.m_primal, when_less.m_primal, when_not_less.m_primal);
  Underlying new_derivative1 = this->m_primal.if_less(
      comparand.m_primal, when_less.m_derivative1, when_not_less.m_derivative1);
  Underlying new_derivative2 = this->m_primal.if_less(
      comparand.m_primal, when_less.m_derivative2, when_not_less.m_derivative2);
  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying>
bool Num3<Underlying>::is_definitely_zero() const {
  return this->m_primal.is_definitely_zero() &&
      this->m_derivative1.is_definitely_zero() &&
      this->m_derivative2.is_definitely_zero();
}

template <class Underlying>
requires Number<Underlying>
bool Num3<Underlying>::is_definitely_one() const {
  return this->m_primal.is_definitely_one() &&
      this->m_derivative1.is_definitely_zero() &&
      this->m_derivative2.is_definitely_zero();
}

static_assert(Number<Num3<Real>>);

} // namespace beanmachine::minibmg
