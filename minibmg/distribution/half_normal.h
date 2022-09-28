/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <random>
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/distribution/distribution.h"
#include "beanmachine/minibmg/distribution/log_transform.h"

namespace beanmachine::minibmg {

template <class Underlying>
requires Number<Underlying>
class HalfNormal : public Distribution<Underlying> {
 private:
  Underlying stddev;

 public:
  explicit HalfNormal(const Underlying& stddev) : stddev{stddev} {}
  double sample(std::mt19937& gen) const override {
    std::normal_distribution<double> dist(0.0, stddev.as_double());
    return std::abs(dist(gen));
  }
  Underlying log_prob(const Underlying& value) const override {
    auto sum_xsq = value * value;
    return -log(value) - std::log(M_PI / 2.0) / 2 -
        0.5 * sum_xsq / (stddev * stddev);
  }
  bool is_discrete() const override {
    return false;
  }
  TransformationPtr<Underlying> transformation() const override {
    return LogTransformation<Underlying>::instance();
  }
};

} // namespace beanmachine::minibmg
