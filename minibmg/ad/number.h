/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <concepts>
#include <string>

namespace beanmachine::minibmg {

// The AD (Auto-Differentiation) facilities are built on the concept of
// a Number.  A type T satisfies the concept Number<T> if it supports
// all of the following operations.
template <typename T>
concept Number = requires(
    const T& a,
    const T& b,
    const T& c,
    const T& d,
    int n,
    double dbl) {
  // there should be a conversion from double to T.
  std::convertible_to<double, T>;
  // it should support the arithmetic oeprators.
  { a + b } -> std::convertible_to<T>;
  { a - b } -> std::convertible_to<T>;
  { -a } -> std::convertible_to<T>;
  { (a * b) } -> std::convertible_to<T>;
  { a / b } -> std::convertible_to<T>;
  // it should support the following transcendental operations.
  { pow(a, b) } -> std::convertible_to<T>;
  { exp(a) } -> std::convertible_to<T>;
  { log(a) } -> std::convertible_to<T>;
  { atan(a) } -> std::convertible_to<T>;
  { lgamma(a) } -> std::convertible_to<T>;
  { polygamma(n, a) } -> std::convertible_to<T>;
  // conditional for equality, less-than.
  { if_equal(a, b, c, d) } -> std::convertible_to<T>;
  { if_less(a, b, c, d) } -> std::convertible_to<T>;
  // There should be a conservative (meaning it may return false even when
  // a number satisfies the test) way to sometimes know if a value is exactly
  // a constant.
  { is_constant(a, dbl) } -> std::same_as<bool>;
  { to_string(a) } -> std::same_as<std::string>;
};

template <typename T>
requires Number<T> T operator+(double l, const T& r) {
  return T{l} + r;
}

template <typename T>
requires Number<T> T operator+(const T& l, double r) {
  return l + T{r};
}

template <typename T>
requires Number<T> T operator-(double l, const T& r) {
  return T{l} - r;
}

template <typename T>
requires Number<T> T operator-(const T& l, double r) {
  return l - T{r};
}

template <typename T>
requires Number<T> T operator*(double l, const T& r) {
  return T{l} * r;
}

template <typename T>
requires Number<T> T operator*(const T& l, double r) {
  return l * T{r};
}

template <typename T>
requires Number<T> T operator/(double l, const T& r) {
  return T{l} / r;
}

template <typename T>
requires Number<T> T operator/(const T& l, double r) {
  return l / T{r};
}

template <typename T>
requires Number<T> T pow(double l, const T& r) {
  return pow(T{l}, r);
}

template <typename T>
requires Number<T> T pow(const T& l, double r) {
  return pow(l, T{r});
}

} // namespace beanmachine::minibmg
