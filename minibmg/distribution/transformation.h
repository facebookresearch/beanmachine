/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include "beanmachine/minibmg/ad/number.h"

namespace beanmachine::minibmg {

// A Transformation is used to map a value from one distribution to another.  We
// assume that all transformations are monotonically increasing (have a
// positive derivative).
//
// References:
//   Change of Variables in statistics:
//   https://online.stat.psu.edu/stat414/lesson/22/22.2
//
//   Stan:
//   https://mc-stan.org/docs/2_27/reference-manual/change-of-variables-section.html
template <class Underlying>
requires Number<Underlying>
class Transformation {
 public:
  // Transform a value from a constrained domain to the unconstrained domain
  // over all reals.
  virtual Underlying call(const Underlying& constrained) const = 0;

  // Perform the functional inverse of transform_sample_forward, transforming a
  // value from the unconstrained domain over all reals to the constrained
  // domain.
  virtual Underlying inverse(const Underlying& unconstrained) const = 0;

  // Transform a log-probability value of a sample from the constrained
  // distribution to the log probability of the transformed sample (from the
  // transformed distribution).
  virtual Underlying transform_log_prob(
      const Underlying& constrained,
      const Underlying& log_prob_constrained) const = 0;

  virtual ~Transformation() {}
};

// A (smart) pointer to a Transformation.
template <class T>
requires Number<T>
using TransformationPtr = std::shared_ptr<const Transformation<T>>;

} // namespace beanmachine::minibmg
