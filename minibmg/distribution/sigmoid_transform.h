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

// A transformation from (0 .. 1) to (-INF .. INF).  This is the logit function.
// See also https://en.wikipedia.org/wiki/Logit
//
// y = f(x) = logit(x) = log(x / (1 - x))
// dy/dx = 1 / (x - x^2)
template <class Underlying>
requires Number<Underlying>
class SigmoidTransformation : public Transformation<Underlying> {
 public:
  SigmoidTransformation() {}
  ~SigmoidTransformation() {}

  static TransformationPtr<Underlying> instance();

  // y = f(x) = logit(x) = log(x / (1 - x))
  Underlying call(const Underlying& constrained) const override {
    return log(constrained / (1 - constrained));
  }

  // The inverse of the logit function is the expit function.
  // See also
  // https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html
  //
  // x = f^{-1}(y) = expit(y) = 1 / (1 + exp(-y))
  Underlying inverse(const Underlying& unconstrained) const override {
    return 1 / (1 + exp(-unconstrained));
  }

  // dy/dx = 1 / (x - x^2)
  // log(dy/dx) = log(1 / (x - x^2)) = -log(x - x^2)
  Underlying transform_log_prob(
      const Underlying& constrained,
      const Underlying& log_prob_constrained) const override {
    // gradient of the transform function
    return log_prob_constrained - log(constrained - constrained * constrained);
  }
};

template <class Underlying>
requires Number<Underlying> TransformationPtr<Underlying>
SigmoidTransformation<Underlying>::instance() {
  static const auto result =
      std::make_shared<const SigmoidTransformation<Underlying>>();
  return result;
}

} // namespace beanmachine::minibmg
