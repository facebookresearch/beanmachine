/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <random>
#include "beanmachine/minibmg/ad/number.h"

namespace beanmachine::minibmg::distribution {

template <class Underlying>
requires Number<Underlying>
class Distribution {
 public:
  virtual double sample(std::mt19937& gen) const;
  virtual Underlying log_prob(const Underlying& value) const;
  virtual ~Distribution() {}
};

} // namespace beanmachine::minibmg::distribution
