/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <random>
#include "beanmachine/minibmg/distribution/distribution.h"

namespace beanmachine::minibmg {

template <class Underlying>
requires Number<Underlying>
class Normal : public Distribution<Underlying> {
 private:
  Underlying mean;
  Underlying stddev;

 public:
  Normal(const Underlying& mean, const Underlying& stddev)
      : mean{mean}, stddev{stddev} {}
  double sample(std::mt19937& gen) const override {
    std::normal_distribution<double> d{mean.as_double(), stddev.as_double()};
    return d(gen);
  }
  Underlying log_prob(const Underlying& value) const override {
    // log[PDF[NormalDistribution[m, s], v]]
    // = -(v-m)^2 / 2s^2 - log(s * sqrt(2 * PI))
    // = -(v-m)^2 / 2s^2 - log(s) - log(sqrt(2 * PI))
    static const double ls2pi = std::log(std::sqrt(2 * M_PI));
    auto vmm = value - mean;
    auto t2 = vmm * vmm;
    return -t2 / (2 * stddev * stddev) - log(stddev) - ls2pi;
  }
  bool is_discrete() const override {
    return false;
  }
  TransformationPtr<Underlying> transformation() const override {
    return nullptr;
  }
};

} // namespace beanmachine::minibmg
