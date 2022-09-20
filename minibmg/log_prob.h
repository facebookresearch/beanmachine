/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>
#include "beanmachine/minibmg/ad/number.h"
#include "beanmachine/minibmg/distribution/bernoulli.h"
#include "beanmachine/minibmg/distribution/beta.h"
#include "beanmachine/minibmg/distribution/normal.h"
#include "beanmachine/minibmg/eval.h"
#include "beanmachine/minibmg/minibmg.h"

namespace beanmachine::minibmg {

using namespace beanmachine::minibmg::distribution;

// Compute the log probability of the given sample being generated
// by the given distribution with the given parameters.
template <class N>
requires Number<N> N
log_prob(Operator distribution, N v, std::function<N(unsigned)> get_parameter) {
  switch (distribution) {
    case Operator::DISTRIBUTION_NORMAL: {
      N mean = get_parameter(0);
      N stddev = get_parameter(1);
      return Normal<N>{mean, stddev}.log_prob(v);
    }
    case Operator::DISTRIBUTION_BETA: {
      N a = get_parameter(0);
      N b = get_parameter(1);
      return Beta<N>{a, b}.log_prob(v);
    }
    case Operator::DISTRIBUTION_BERNOULLI: {
      N probability_of_one = get_parameter(0);
      return Bernoulli<N>{probability_of_one}.log_prob(v);
    }
    default:
      throw EvalError("log_prob does not support " + to_string(distribution));
  }
}

} // namespace beanmachine::minibmg
