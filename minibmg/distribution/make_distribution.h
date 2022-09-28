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
#include "beanmachine/minibmg/distribution/half_normal.h"
#include "beanmachine/minibmg/distribution/normal.h"
#include "beanmachine/minibmg/eval_error.h"
#include "beanmachine/minibmg/operator.h"

namespace beanmachine::minibmg {

// Create a distribution object for the given distribution operator and
// parameters.
template <class N>
requires Number<N> DistributionPtr<N> make_distribution(
    Operator distribution,
    std::function<N(unsigned)> get_parameter) {
  switch (distribution) {
    case Operator::DISTRIBUTION_HALF_NORMAL: {
      N stddev = get_parameter(0);
      return std::make_shared<const HalfNormal<N>>(stddev);
    }
    case Operator::DISTRIBUTION_NORMAL: {
      N mean = get_parameter(0);
      N stddev = get_parameter(1);
      return std::make_shared<Normal<N>>(mean, stddev);
    }
    case Operator::DISTRIBUTION_BETA: {
      N a = get_parameter(0);
      N b = get_parameter(1);
      return std::make_shared<Beta<N>>(a, b);
    }
    case Operator::DISTRIBUTION_BERNOULLI: {
      N probability_of_one = get_parameter(0);
      return std::make_shared<Bernoulli<N>>(probability_of_one);
    }
    default:
      throw EvalError(
          "make_distribution does not support " + to_string(distribution));
  }
}

} // namespace beanmachine::minibmg
