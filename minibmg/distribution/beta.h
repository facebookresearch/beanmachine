/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <random>
#include "beanmachine/minibmg/distribution/distribution.h"
#include "beanmachine/minibmg/eval.h"

namespace beanmachine::minibmg::distribution {

template <class Underlying>
requires Number<Underlying>
class Beta : public Distribution<Underlying> {
 public:
  const Underlying a;
  const Underlying b;
  Beta(const Underlying& a, const Underlying& b) : a{a}, b{b} {}
  double sample(std::mt19937& gen) const override {
    std::gamma_distribution<double> distrib_a(a.as_double(), 1);
    std::gamma_distribution<double> distrib_b(b.as_double(), 1);
    double x = distrib_a(gen);
    double y = distrib_b(gen);
    double sum = x + y;
    if (sum == 0.0) {
      throw EvalError("sample_distribution has a degenerate Beta");
    }
    double p = x / sum;
    return p;
  }
  Underlying log_prob(const Underlying& value) const override {
    return (a - 1) * value.log() + (b - 1) * (1 - value).log() +
        (a + b).lgamma() - a.lgamma() - b.lgamma();
  }
};

} // namespace beanmachine::minibmg::distribution
