/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

namespace beanmachine::minibmg {

enum class Operator {
  // An operator value that indicates no operator.  Used as a flag to
  // reflect an invalid operator value.
  NO_OPERATOR,

  // A scalar constant, like 1.2.
  // Result: the given constant value (REAL)
  CONSTANT,

  // A scalar variable.  Used for symbolid auto-differentiation (AD).
  VARIABLE,

  // Add two scalars.
  // Result: the sum (REAL)
  ADD,

  // Subtract one scalar from another.
  // Result: the difference (REAL)
  SUBTRACT,

  // Negate a scalar.
  // Result: the negated value (REAL)
  NEGATE,

  // Multiply two scalars.
  // Result: the product (REAL)
  MULTIPLY,

  // Divide one scalar by another.
  // Result: the quotient (REAL)
  DIVIDE,

  // Raise on scalar to the power of another.
  // Result: REAL
  POW,

  // Raise e to the power of the given scalar.
  // Result: REAL
  EXP,

  // The natural logarithm of the given scalar.
  // Result: REAL
  LOG,

  // The arctangent (functional inverse of the tangent) of a scalar.
  // Result: REAL
  ATAN,

  // The lgamma function
  // Result: REAL
  LGAMMA,

  // The polygamma(x, n) function.  polygamma(x, 0) is also known as digamma(x)
  // Note the order of parameters.
  // Result: REAL
  POLYGAMMA,

  // If the first argument is equal to the second, yields the third, else the
  // fourth.
  // Result: REAL
  IF_EQUAL,

  // If the first argument is less than the second, yields the third, else the
  // fourth.
  // Result: REAL
  IF_LESS,

  // A normal distribution.
  // Parameters:
  // - mean (REAL)
  // - standard deviation (REAL)
  // Result: the distribution (DISTRIBUTION)
  DISTRIBUTION_NORMAL,

  // A beta distribution.
  // Parameters:
  // - ??? (REAL)
  // - ??? (REAL)
  // Result: the distribution.
  DISTRIBUTION_BETA,

  // A bernoulli distribution (DISTRIBUTION).
  // Parameters:
  // - probability of yeilding 1 (as opposed to 0) (REAL)
  // Result: the distribution (DISTRIBUTION)
  DISTRIBUTION_BERNOULLI,

  // Draw a sample from the distribution parameter.
  // Parameters:
  // - ditribution
  // Result: A sample from the distribution (REAL)
  SAMPLE,

  // Not a real operator.  Used as a limit when looping through operators.
  LAST_OPERATOR,
};

Operator operator_from_name(const std::string& name);
std::string to_string(Operator op);
unsigned int arity(Operator op);

} // namespace beanmachine::minibmg
