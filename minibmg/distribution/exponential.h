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
class Exponential : public Distribution<Underlying> {
 private:
  Underlying rate;

 public:
  explicit Exponential(const Underlying& rate) : rate{rate} {}
  double sample(std::mt19937& gen) const override {
    std::exponential_distribution<double> dist(rate.as_double());
    return dist(gen);
  }
  Underlying log_prob(const Underlying& value) const override {
    return log(rate) - rate * value;
  }
  bool is_discrete() const override {
    return false;
  }
  TransformationPtr<Underlying> transformation() const override {
    return LogTransformation<Underlying>::instance();
  }
};

} // namespace beanmachine::minibmg
