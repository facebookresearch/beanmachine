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

class Gamma : public Proposer {
 public:
  /*
  Constructor for Gamma class.
  pdf(x) = x^(alpha - 1) e^{-beta * x}  * beta^alpha / Gamma-func(alpha)  for x
  >= 0 :param alpha: shape :param beta: rate
  */
  Gamma(double alpha, double beta) : Proposer(), alpha(alpha), beta(beta) {}
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
  double alpha;
  double beta;
};

} // namespace proposer
} // namespace beanmachine
