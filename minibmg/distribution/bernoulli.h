/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <random>
#include "beanmachine/minibmg/distribution/distribution.h"

namespace beanmachine::minibmg {

template <class Underlying>
requires Number<Underlying>
class Bernoulli : public Distribution<Underlying> {
 public:
  const Underlying probability_of_one;
  double sample(std::mt19937& gen) const override {
    std::uniform_real_distribution<double> p(0.0, 1.0);
    bool result = p(gen) < probability_of_one.as_double();
    return double(result);
  }
  Underlying log_prob(const Underlying& value) const override {
    auto probability_of_zero = 1 - probability_of_one;
    return value.if_equal(
        1,
        log(probability_of_one),
        value.if_equal(0, log(probability_of_zero), -INFINITY));
  }
  bool is_discrete() const override {
    return false;
  }
};

} // namespace beanmachine::minibmg
