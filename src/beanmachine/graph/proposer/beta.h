/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "beanmachine/graph/proposer/proposer.h"

namespace beanmachine {
namespace proposer {

class Beta : public Proposer {
 public:
  /*
  Constructor for Beta class.
  :param a: shape param for Beta
  :param b: shape param for Beta
  */
  Beta(double a, double b) : Proposer(), a(a), b(b) {}
  /*
  Sample a value from the proposer.
  :param gen: Random number generator.
  :returns: A value.
  */
  graph::NodeValue sample(std::mt19937& gen) const override;
  /*
  Compute the log_prob of a value.
  :param value: The value to evaluate the distribution.
  :returns: log probability of value.
  */
  double log_prob(graph::NodeValue& value) const override;

 private:
  double a;
  double b;
};

} // namespace proposer
} // namespace beanmachine
