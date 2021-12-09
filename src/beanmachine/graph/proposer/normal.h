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

class Normal : public Proposer {
 public:
  /*
  Constructor for Normal class.
  :param mu: mean of Normal
  :param sigma: std. dev of Normal
  */
  Normal(double mu, double sigma) : Proposer(), mu(mu), sigma(sigma) {}
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
  double mu;
  double sigma;
};

} // namespace proposer
} // namespace beanmachine
