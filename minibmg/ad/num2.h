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
#include <unordered_map>
#include <vector>
#include "beanmachine/minibmg/ad/number.h"
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/dedup.h"

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
  Underlying primal, derivative1;

  /* implicit */ Num2(double primal);
  /* implicit */ Num2(Underlying primal);
  Num2(Underlying primal, Underlying derivative1);
  Num2();
  Num2(const Num2<Underlying>& other);
  Num2<Underlying>& operator=(const Num2<Underlying>& other) = default;
  double as_double() const;
};

template <class Underlying>
requires Number<Underlying> Num2<Underlying>::Num2()
    : primal{0}, derivative1{0} {}

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
requires Number<Underlying> Num2<Underlying>::Num2(
    const Num2<Underlying>& other)
    : primal(other.primal), derivative1(other.derivative1) {}

template <class Underlying>
requires Number<Underlying>
double Num2<Underlying>::as_double() const {
  return primal.as_double();
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying>
operator+(const Num2<Underlying>& left, const Num2<Underlying>& right) {
  return Num2<Underlying>{
      left.primal + right.primal, left.derivative1 + right.derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying>
operator-(const Num2<Underlying>& left, const Num2<Underlying>& right) {
  return Num2<Underlying>{
      left.primal - right.primal, left.derivative1 - right.derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying>
operator-(const Num2<Underlying>& x) {
  return Num2<Underlying>{-x.primal, -x.derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying>
operator*(const Num2<Underlying>& left, const Num2<Underlying>& right) {
  return Num2<Underlying>{
      left.primal * right.primal,
      left.primal * right.derivative1 + left.derivative1 * right.primal};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying>
operator/(const Num2<Underlying>& left, const Num2<Underlying>& right) {
  // a / b
  Underlying new_primal = left.primal / right.primal;

  // Using https://www.wolframalpha.com/
  // D[a(x) / b(x), x] -> (b[x] a'[x] - a[x] b'[x])/b[x]^2
  auto other2 = right.primal * right.primal;
  Underlying new_derivative1 =
      (right.primal * left.derivative1 - left.primal * right.derivative1) /
      other2;
  return Num2<Underlying>{new_primal, new_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> pow(
    const Num2<Underlying>& base,
    const Num2<Underlying>& exponent) {
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
  //        = a[x]^(b[x] - 1) (b[x] a'[x] + a[x] Log[a[x]] b'[x])

  // We avoid using the log (e.g. of a negative number) when not needed.
  // t0 = a[x] Log[a[x]] b'[x]
  const Underlying t0 =
      is_constant(exponent.derivative1, 0) || is_constant(base.primal, 0)
      ? 0
      : base.primal * log(base.primal) * exponent.derivative1;

  // b[x] * a'[x] + a[x] Log[a[x]] b'[x]
  const Underlying x0 = exponent.primal * base.derivative1 + t0;

  // a[x]^(b[x] - 1) (b[x] a'[x] + a[x] Log[a[x]] b'[x])
  Underlying new_derivative1 = x0 * pow(base.primal, exponent.primal - 1);

  return Num2<Underlying>{new_primal, new_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> exp(const Num2<Underlying>& x) {
  Underlying new_primal = exp(x.primal);
  Underlying new_derivative1 = new_primal * x.derivative1;
  return Num2<Underlying>{new_primal, new_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> log(const Num2<Underlying>& x) {
  Underlying new_primal = log(x.primal);
  Underlying new_derivative1 = x.derivative1 / x.primal;
  return Num2<Underlying>{new_primal, new_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> atan(const Num2<Underlying>& x) {
  Underlying new_primal = atan(x.primal);
  Underlying new_derivative1 = x.derivative1 / (x.primal * x.primal + 1.0f);
  return Num2<Underlying>{new_primal, new_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> lgamma(const Num2<Underlying>& x) {
  Underlying new_primal = lgamma(x.primal);
  Underlying new_derivative1 = x.derivative1 * polygamma(0, x.primal);
  return Num2<Underlying>{new_primal, new_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> polygamma(
    int n,
    const Num2<Underlying>& x) {
  Underlying new_primal = polygamma(n, x.primal);
  Underlying new_derivative1 = x.derivative1 * polygamma(n + 1, x.primal);
  return Num2<Underlying>{new_primal, new_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> log1p(const Num2<Underlying>& x) {
  Underlying new_primal = log1p(x.primal);
  // f = log(1 + x)
  // f' = x' / (1 + x)
  Underlying new_derivative1 = x.derivative1 / (1 + x.primal);
  return Num2<Underlying>{new_primal, new_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> if_equal(
    const Num2<Underlying>& value,
    const Num2<Underlying>& comparand,
    const Num2<Underlying>& when_equal,
    const Num2<Underlying>& when_not_equal) {
  // Note: we discard and ignore left.derivative1 and
  // comparand->derivative1
  Underlying new_primal = if_equal(
      value.primal, comparand.primal, when_equal.primal, when_not_equal.primal);
  Underlying new_derivative1 = if_equal(
      value.primal,
      comparand.primal,
      when_equal.derivative1,
      when_not_equal.derivative1);
  return Num2<Underlying>{new_primal, new_derivative1};
}

template <class Underlying>
requires Number<Underlying> Num2<Underlying> if_less(
    const Num2<Underlying>& value,
    const Num2<Underlying>& comparand,
    const Num2<Underlying>& when_less,
    const Num2<Underlying>& when_not_less) {
  // Note: we discard and ignore left.derivative1 and
  // comparand->derivative1
  Underlying new_primal = if_less(
      value.primal, comparand.primal, when_less.primal, when_not_less.primal);
  Underlying new_derivative1 = if_less(
      value.primal,
      comparand.primal,
      when_less.derivative1,
      when_not_less.derivative1);
  return Num2<Underlying>{new_primal, new_derivative1};
}

template <class Underlying>
requires Number<Underlying>
bool is_constant(const Num2<Underlying>& x, double& value) {
  return is_constant(x.derivative1, 0) && is_constant(x.primal, value);
}

template <class Underlying>
requires Number<Underlying>
bool is_constant(const Num2<Underlying>& x, const double& value) {
  return is_constant(x.derivative1, 0) && is_constant(x.primal, value);
}

template <class Underlying>
requires Number<Underlying> std::string to_string(const Num2<Underlying>& x) {
  return fmt::format(
      "[primal={0}, derivative={1}]",
      to_string(x.primal),
      to_string(x.derivative1));
}

static_assert(Number<Num2<Real>>);

template <class Underlying>
requires Number<Underlying>
class DedupAdapter<Num2<Underlying>> {
  DedupAdapter<Underlying> helper{};

 public:
  std::vector<Nodep> find_roots(const Num2<Underlying>& num2) const {
    std::vector<Nodep> result = helper.find_roots(num2.primal);
    for (auto& n : helper.find_roots(num2.derivative1)) {
      result.push_back(n);
    }
    return result;
  }
  Num2<Underlying> rewrite(
      const Num2<Underlying>& num2,
      const std::unordered_map<Nodep, Nodep>& map) const {
    auto new_primal = helper.rewrite(num2.primal, map);
    auto new_derivative1 = helper.rewrite(num2.derivative1, map);
    return {new_primal, new_derivative1};
  }
};

} // namespace beanmachine::minibmg
