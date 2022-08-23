/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/format.h>
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
  bool is_constant(double& value) const;
  bool is_constant(const double& value) const;
  std::string to_string() const;
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
  // a / b
  Underlying new_primal = this->m_primal / other.m_primal;

  // Using https://www.wolframalpha.com/
  // D[a(x) / b(x), x] -> (b[x] a'[x] - a[x] b'[x])/b[x]^2
  auto other2 = other.m_primal * other.m_primal;
  Underlying new_derivative1 = (other.m_primal * this->m_derivative1 -
                                this->m_primal * other.m_derivative1) /
      other2;

  // D[a(x) / b(x), {x, 2}] ->
  //     (2 a[x] b'[x]^2 +
  //        b[x]^2 a''[x] -
  //        b[x] (2 a'[x] b'[x] + a[x] b''[x]))
  //              / b[x]^3
  auto other3 = other2 * other.m_primal;
  Underlying new_derivative2 =
      (2 * this->m_primal * other.m_derivative1 * other.m_derivative1 +
       other2 * this->m_derivative2 -
       other.m_primal *
           (2 * this->m_derivative1 * other.m_derivative1 +
            this->m_primal * other.m_derivative2)) /
      other3;

  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::pow(
    const Num3<Underlying>& other)
const {
  double power;
  // shortcut some cases.
  if (this->is_constant(power)) {
    if (power == 0)
      return 1;
    if (power == 1)
      return *this;
  }

  const Underlying new_primal = this->m_primal.pow(other.m_primal);

  // From https://www.wolframalpha.com/
  // D[a(x) ^ (b(x)), x] ->
  //          a[x]^(-1 + b[x]) (b[x] a'[x] + a[x] Log[a[x]] b'[x])
  //        = a[x]^b[x] (b[x] a'[x] / a[x] + Log[a[x]] b'[x])

  // avoid using the log (e.g. of a negative number) when not needed.
  const Underlying loga = this->m_primal.log();

  // Log[a[x]] b'[x]
  const Underlying t0 =
      other.m_derivative1.is_constant(0) ? 0 : loga * other.m_derivative1;

  // b[x] * a'[x] / a[x] + Log[a[x]] b'[x]
  const Underlying x0 =
      other.m_primal * this->m_derivative1 / this->m_primal + t0;

  // a[x]^b[x] (b[x] a'[x] / a[x] + Log[a[x]] b'[x])
  Underlying new_derivative1 = new_primal * x0;

  // D[a(x) ^ (b(x)), {x, 2}] ->
  //          a[x]^b[x] ((b[x] a'[x])/a[x] + Log[a[x]] b'[x])^2 -
  //                     ((b[x] a'[x]^2)/a[x]^2) +
  //                     (2 a'[x] b'[x])/a[x] +
  //                     (b[x] a''[x])/a[x] +
  //                     Log[a[x]] b''[x])

  // ((b[x] a'[x])/a[x] + Log[a[x]] b'[x])^2
  const Underlying t1 = x0.pow(2);

  // -((b[x] a'[x]^2)/a[x]^2)
  const Underlying t2 =
      -(other.m_primal * this->m_derivative1.pow(2) / this->m_primal.pow(2));

  // (2 a'[x] b'[x])/a[x]
  const Underlying t3 =
      2 * other.m_derivative1 * this->m_derivative1 / this->m_primal;

  // (b[x] a''[x])/a[x]
  const Underlying t4 = other.m_primal * this->m_derivative2 / this->m_primal;

  // Log[a[x]] b''[x]
  const Underlying t5 =
      other.m_derivative2.is_constant(0) ? 0 : loga * other.m_derivative2;

  const Underlying new_derivative2 = new_primal * (t1 + t2 + t3 + t4 + t5);

  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> Num3<Underlying>::exp()
const {
  const Underlying new_primal = this->m_primal.exp();
  const Underlying new_derivative1 = new_primal * this->m_derivative1;
  const Underlying new_derivative2 = (new_primal * this->m_derivative2) +
      (this->m_derivative1 * new_derivative1);
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
  const Underlying new_primal = this->m_primal.atan();

  // D[ArcTan[f[x]], x] ->
  //        f'[x]/(1 + f[x]^2)

  // (1 + f[x]^2)
  auto t1 = 1 + this->m_primal.pow(2);
  // f'[x]/(1 + f[x]^2)
  const Underlying new_derivative1 = this->m_derivative1 / t1;

  // D[ArcTan[f[x]], {x, 2}] ->
  //        (-2 f[x] f'[x]^2 + (1 + f[x]^2) f''[x])/(1 + f[x]^2)^2
  const Underlying new_derivative2 =
      (-2 * this->m_primal * this->m_derivative1.pow(2) +
       t1 * this->m_derivative2) /
      t1.pow(2);

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
bool Num3<Underlying>::is_constant(double& value) const {
  return this->m_derivative1.is_constant(0) &&
      this->m_primal.is_constant(value);
}

template <class Underlying>
requires Number<Underlying>
bool Num3<Underlying>::is_constant(const double& value) const {
  return this->m_derivative1.is_constant(0) &&
      this->m_primal.is_constant(value);
}

template <class Underlying>
requires Number<Underlying> std::string Num3<Underlying>::to_string()
const {
  return fmt::format(
      "[primal={0}, derivative={1}, derivative2={2}]",
      this->m_primal.to_string(),
      this->m_derivative1.to_string(),
      this->m_derivative2.to_string());
}

static_assert(Number<Num3<Real>>);

} // namespace beanmachine::minibmg
