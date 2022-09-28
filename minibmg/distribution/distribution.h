/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <random>
#include "beanmachine/minibmg/ad/number.h"
#include "beanmachine/minibmg/distribution/transformation.h"

namespace beanmachine::minibmg {

template <class Underlying>
requires Number<Underlying>
class Distribution;

// A (smart) pointer to a Distribution.
template <class T>
requires Number<T>
using DistributionPtr = std::shared_ptr<const Distribution<T>>;

// A distribution of random values, which can be used to produce samples or tell
// us the log of the value of its probability distribution function at a value.
template <class Underlying>
requires Number<Underlying>
class Distribution {
 public:
  // Generate a sample from the distribution.
  virtual double sample(std::mt19937& gen) const = 0;
  // Compute the log of the probability distribution function for the
  // distribution evaluated at a particular value.
  virtual Underlying log_prob(const Underlying& value) const = 0;
  // return true if and only if the distribution support set is discrete (rather
  // than continuous).
  virtual bool is_discrete() const = 0;
  virtual ~Distribution() {}

  // A transformation that maps this distribution to a distribution over (-INF
  // .. INF).  This is nullptr if no transformation is required: for example,
  // for a normal distribution that is already over (-INF .. INF).
  virtual TransformationPtr<Underlying> transformation() const = 0;
};

} // namespace beanmachine::minibmg
