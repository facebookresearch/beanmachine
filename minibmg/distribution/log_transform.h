/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include "beanmachine/minibmg/ad/num2.h"
#include "beanmachine/minibmg/distribution/transformation.h"

namespace beanmachine::minibmg {

// A transformation from (0 .. INF) to (-INF .. INF).  This is the natural
// logarithm.
template <class Underlying>
requires Number<Underlying>
class LogTransformation : public Transformation<Underlying> {
 public:
  LogTransformation() {}
  ~LogTransformation() {}

  static TransformationPtr<Underlying> instance();

  Underlying call(const Underlying& constrained) const override {
    return log(constrained);
  }

  Underlying inverse(const Underlying& unconstrained) const override {
    return exp(unconstrained);
  }

  // dy/dx = 1 / x
  // log(dy/dx) = -log(x)
  Underlying transform_log_prob(
      const Underlying& constrained,
      const Underlying& log_prob_constrained) const override {
    // gradient of the transform function
    return log_prob_constrained - log(constrained);
  }
};

template <class Underlying>
requires Number<Underlying> TransformationPtr<Underlying>
LogTransformation<Underlying>::instance() {
  static const auto result = std::make_shared<LogTransformation<Underlying>>();
  return result;
}

} // namespace beanmachine::minibmg
