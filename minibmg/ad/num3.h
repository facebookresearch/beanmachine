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
#include <vector>
#include "beanmachine/minibmg/ad/number.h"
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/dedup.h"
#include "beanmachine/minibmg/dedup2.h"
#include "beanmachine/minibmg/node.h"

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
  Underlying primal;
  Underlying derivative1;
  Underlying derivative2;

  /* implicit */ Num3(double primal);
  /* implicit */ Num3(Underlying primal);
  Num3();
  Num3(Underlying primal, Underlying derivative1, Underlying derivative2);
  Num3(const Num3<Underlying>& other);
  Num3<Underlying>& operator=(const Num3<Underlying>& other) = default;
  double as_double() const;
};

template <class Underlying>
requires Number<Underlying> Num3<Underlying>::Num3()
    : primal{0}, derivative1{0}, derivative2{0} {}

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
requires Number<Underlying> Num3<Underlying>::Num3(
    const Num3<Underlying>& other)
    : primal(other.primal),
      derivative1(other.derivative1),
      derivative2(other.derivative2) {}

template <class Underlying>
requires Number<Underlying>
double Num3<Underlying>::as_double() const {
  return this->primal.as_double();
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying>
operator+(const Num3<Underlying>& left, const Num3<Underlying>& right) {
  return Num3<Underlying>{
      left.primal + right.primal,
      left.derivative1 + right.derivative1,
      left.derivative2 + right.derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying>
operator-(const Num3<Underlying>& left, const Num3<Underlying>& right) {
  return Num3<Underlying>{
      left.primal - right.primal,
      left.derivative1 - right.derivative1,
      left.derivative2 - right.derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying>
operator-(const Num3<Underlying>& x) {
  return Num3<Underlying>{-x.primal, -x.derivative1, -x.derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying>
operator*(const Num3<Underlying>& left, const Num3<Underlying>& right) {
  // a * b
  Underlying new_primal = left.primal * right.primal;
  // Derivative: d/dx (a * b)
  //             = a' * b + b' * a (product rule)
  Underlying new_derivative1 =
      left.derivative1 * right.primal + right.derivative1 * left.primal;
  // Second derivative: d/dx d/dx (a * b)
  //             = d/dx (a' * b + a * b') (product rule)
  //             = d/dx (a' * b) + d/dx (a * b') // sum rule
  //             = (a'' * b + a' * b') + (a' * b' + a * b'') // product rule
  //             = a'' * b + 2 * a' * b' + a * b'' // gather terms
  Underlying new_derivative2 = left.derivative2 * right.primal +
      2 * left.derivative1 * right.derivative1 +
      left.primal * right.derivative2;
  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying>
operator/(const Num3<Underlying>& left, const Num3<Underlying>& right) {
  // a / b
  Underlying new_primal = left.primal / right.primal;

  // Using https://www.wolframalpha.com/
  // D[a(x) / b(x), x] -> (b[x] a'[x] - a[x] b'[x])/b[x]^2
  auto other2 = right.primal * right.primal;
  Underlying new_derivative1 =
      (right.primal * left.derivative1 - left.primal * right.derivative1) /
      other2;

  // D[a(x) / b(x), {x, 2}] ->
  //     (2 a[x] b'[x]^2 +
  //        b[x]^2 a''[x] -
  //        b[x] (2 a'[x] b'[x] + a[x] b''[x]))
  //              / b[x]^3
  auto other3 = other2 * right.primal;
  Underlying new_derivative2 =
      (2 * left.primal * right.derivative1 * right.derivative1 +
       other2 * left.derivative2 -
       right.primal *
           (2 * left.derivative1 * right.derivative1 +
            left.primal * right.derivative2)) /
      other3;

  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> pow(
    const Num3<Underlying>& base,
    const Num3<Underlying>& exponent) {
  double power;
  // shortcut some cases.
  if (is_constant(exponent, power)) {
    if (power == 0)
      return 1;
    if (power == 1)
      return base;
  }

  const Underlying new_primal = pow(base.primal, exponent.primal);

  // From https://www.wolframalpha.com/
  // D[a(x) ^ (b(x)), x] ->
  //          a[x]^(-1 + b[x]) (b[x] a'[x] + a[x] Log[a[x]] b'[x])
  //        = a[x]^b[x] (b[x] a'[x] / a[x] + Log[a[x]] b'[x])

  // avoid using the log (e.g. of a negative number) when not needed.
  const Underlying loga = log(base.primal);

  // Log[a[x]] b'[x]
  const Underlying t0 =
      is_constant(exponent.derivative1, 0) ? 0 : loga * exponent.derivative1;

  // b[x] * a'[x] / a[x] + Log[a[x]] b'[x]
  const Underlying x0 = exponent.primal * base.derivative1 / base.primal + t0;

  // a[x]^b[x] (b[x] a'[x] / a[x] + Log[a[x]] b'[x])
  Underlying new_derivative1 = new_primal * x0;

  // D[a(x) ^ (b(x)), {x, 2}] ->
  //          a[x]^b[x] ((b[x] a'[x])/a[x] + Log[a[x]] b'[x])^2 -
  //                     ((b[x] a'[x]^2)/a[x]^2) +
  //                     (2 a'[x] b'[x])/a[x] +
  //                     (b[x] a''[x])/a[x] +
  //                     Log[a[x]] b''[x])

  // ((b[x] a'[x])/a[x] + Log[a[x]] b'[x])^2
  const Underlying t1 = pow(x0, 2);

  // -((b[x] a'[x]^2)/a[x]^2)
  const Underlying t2 =
      -(exponent.primal * pow(base.derivative1, 2) / pow(base.primal, 2));

  // (2 a'[x] b'[x])/a[x]
  const Underlying t3 =
      2 * exponent.derivative1 * base.derivative1 / base.primal;

  // (b[x] a''[x])/a[x]
  const Underlying t4 = exponent.primal * base.derivative2 / base.primal;

  // Log[a[x]] b''[x]
  const Underlying t5 =
      is_constant(exponent.derivative2, 0) ? 0 : loga * exponent.derivative2;

  const Underlying new_derivative2 = new_primal * (t1 + t2 + t3 + t4 + t5);

  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> exp(const Num3<Underlying>& x) {
  const Underlying new_primal = exp(x.primal);
  const Underlying new_derivative1 = new_primal * x.derivative1;
  const Underlying new_derivative2 =
      (new_primal * x.derivative2) + (x.derivative1 * new_derivative1);
  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> log(const Num3<Underlying>& x) {
  const Underlying new_primal = log(x.primal);
  const Underlying new_derivative1 = x.derivative1 / x.primal;
  const Underlying new_derivative2 = (x.derivative2 / x.primal) -
      (x.derivative1 * (x.derivative1 * pow(x.primal, -2.0f)));
  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> atan(const Num3<Underlying>& x) {
  const Underlying new_primal = atan(x.primal);

  // D[ArcTan[f[x]], x] ->
  //        f'[x]/(1 + f[x]^2)

  // (1 + f[x]^2)
  auto t1 = 1 + pow(x.primal, 2);
  // f'[x]/(1 + f[x]^2)
  const Underlying new_derivative1 = x.derivative1 / t1;

  // D[ArcTan[f[x]], {x, 2}] ->
  //        (-2 f[x] f'[x]^2 + (1 + f[x]^2) f''[x])/(1 + f[x]^2)^2
  const Underlying new_derivative2 =
      (-2 * x.primal * pow(x.derivative1, 2) + t1 * x.derivative2) / pow(t1, 2);

  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> lgamma(const Num3<Underlying>& x) {
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
  Underlying new_primal = lgamma(x.primal);
  auto t1 = polygamma(0, x.primal); // f’(g(x))
  Underlying new_derivative1 = x.derivative1 * t1;
  Underlying new_derivative2 =
      polygamma(1, x.primal) * x.derivative1 * x.derivative1 +
      x.derivative2 * t1;
  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> polygamma(
    int n,
    const Num3<Underlying>& x) {
  Underlying new_primal = polygamma(n, x.primal);
  auto t1 = polygamma(n + 1, x.primal);
  Underlying new_derivative1 = x.derivative1 * t1;
  Underlying new_derivative2 =
      polygamma(n + 2, x.primal) * x.derivative1 * x.derivative1 +
      x.derivative2 * t1;
  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> if_equal(
    const Num3<Underlying>& value,
    const Num3<Underlying>& comparand,
    const Num3<Underlying>& when_equal,
    const Num3<Underlying>& when_not_equal) {
  // Note: we discard and ignore left.derivative* and comparand->derivative*
  Underlying new_primal = if_equal(
      value.primal, comparand.primal, when_equal.primal, when_not_equal.primal);
  Underlying new_derivative1 = if_equal(
      value.primal,
      comparand.primal,
      when_equal.derivative1,
      when_not_equal.derivative1);
  Underlying new_derivative2 = if_equal(
      value.primal,
      comparand.primal,
      when_equal.derivative2,
      when_not_equal.derivative2);
  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying> Num3<Underlying> if_less(
    const Num3<Underlying>& value,
    const Num3<Underlying>& comparand,
    const Num3<Underlying>& when_less,
    const Num3<Underlying>& when_not_less) {
  // Note: we discard and ignore left.derivative* and comparand->derivative*
  Underlying new_primal = if_less(
      value.primal, comparand.primal, when_less.primal, when_not_less.primal);
  Underlying new_derivative1 = if_less(
      value.primal,
      comparand.primal,
      when_less.derivative1,
      when_not_less.derivative1);
  Underlying new_derivative2 = if_less(
      value.primal,
      comparand.primal,
      when_less.derivative2,
      when_not_less.derivative2);
  return Num3<Underlying>{new_primal, new_derivative1, new_derivative2};
}

template <class Underlying>
requires Number<Underlying>
bool is_constant(const Num3<Underlying>& x, double& value) {
  return is_constant(x.derivative1, 0) && is_constant(x.primal, value);
}

template <class Underlying>
requires Number<Underlying>
bool is_constant(const Num3<Underlying>& x, const double& value) {
  return is_constant(x.derivative1, 0) && x.primal.is_constant(value);
}

template <class Underlying>
requires Number<Underlying> std::string to_string(const Num3<Underlying>& x) {
  return fmt::format(
      "[primal={0}, derivative={1}, derivative2={2}]",
      x.primal.to_string(),
      x.derivative1.to_string(),
      x.derivative2.to_string());
}

static_assert(Number<Num3<Real>>);

template <class Underlying>
requires Number<Underlying>
class DedupHelper<Num3<Underlying>> {
  DedupHelper<Underlying> helper{};

 public:
  std::vector<Nodep> find_roots(const Num3<Underlying>& num3) const {
    std::vector<Nodep> result = helper.find_roots(num3.primal);
    for (auto& n : helper.find_roots(num3.derivative1)) {
      result.push_back(n);
    }
    for (auto& n : helper.find_roots(num3.derivative2)) {
      result.push_back(n);
    }
    return result;
  }
  Num3<Underlying> rewrite(
      const Num3<Underlying>& num3,
      const std::unordered_map<Nodep, Nodep>& map) const {
    auto new_primal = helper.rewrite(num3.primal, map);
    auto new_derivative1 = helper.rewrite(num3.derivative1, map);
    auto new_derivative2 = helper.rewrite(num3.derivative2, map);
    return {new_primal, new_derivative1, new_derivative2};
  }
};

template <class Underlying>
requires Number<Underlying>
class DedupAdapter<Num3<Underlying>> {
  DedupAdapter<Underlying> helper{};

 public:
  std::vector<Node2p> find_roots(const Num3<Underlying>& num3) const {
    std::vector<Node2p> result = helper.find_roots(num3.primal);
    for (auto& n : helper.find_roots(num3.derivative1)) {
      result.push_back(n);
    }
    for (auto& n : helper.find_roots(num3.derivative2)) {
      result.push_back(n);
    }
    return result;
  }
  Num3<Underlying> rewrite(
      const Num3<Underlying>& num3,
      const std::unordered_map<Node2p, Node2p>& map) const {
    auto new_primal = helper.rewrite(num3.primal, map);
    auto new_derivative1 = helper.rewrite(num3.derivative1, map);
    auto new_derivative2 = helper.rewrite(num3.derivative2, map);
    return {new_primal, new_derivative1, new_derivative2};
  }
};

} // namespace beanmachine::minibmg
