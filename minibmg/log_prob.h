/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>
#include "beanmachine/minibmg/ad/number.h"
#include "beanmachine/minibmg/eval.h"
#include "beanmachine/minibmg/minibmg.h"

namespace beanmachine::minibmg {

template <class N>
requires Number<N> N log_prob_normal(N v, N mean, N stdev) {
  // log[PDF[NormalDistribution[m, s], v]]
  // = -(v-m)^2 / 2s^2 - log(s * sqrt(2 * PI))
  // = -(v-m)^2 / 2s^2 - log(s) - log(sqrt(2 * PI))
  static auto ls2pi = std::log(std::sqrt(2 * M_PI));
  auto vmm = v - mean;
  auto t2 = vmm * vmm;
  return -t2 / (2 * stdev * stdev) - stdev.log() - ls2pi;
}

template <class N>
requires Number<N> N log_prob_beta(N v, N a, N b) {
  return (a - 1) * v.log() + (b - 1) * (1 - v).log() + (a + b).lgamma() -
      a.lgamma() - b.lgamma();
}

template <class N>
requires Number<N> N log_prob_bernoulli(N v, N probability_of_one) {
  N probability_of_zero = 1 - probability_of_one;
  return v.if_equal(
      1,
      probability_of_one.log(),
      v.if_equal(0, probability_of_zero.log(), -INFINITY));
}

// Compute the log probability of the given sample being generated
// by the given distribution with the given parameters.
template <class N>
requires Number<N> N
log_prob(Operator distribution, N v, std::function<N(uint)> get_parameter) {
  switch (distribution) {
    case Operator::DISTRIBUTION_NORMAL: {
      N m = get_parameter(0);
      N s = get_parameter(1);
      return log_prob_normal(v, m, s);
    }
    case Operator::DISTRIBUTION_BETA: {
      N a = get_parameter(0);
      N b = get_parameter(1);
      return log_prob_beta(v, a, b);
    }
    case Operator::DISTRIBUTION_BERNOULLI: {
      N probability_of_one = get_parameter(0);
      return log_prob_bernoulli(v, probability_of_one);
    }
    default:
      throw EvalError("log_prob does not support " + to_string(distribution));
  }
}

} // namespace beanmachine::minibmg
