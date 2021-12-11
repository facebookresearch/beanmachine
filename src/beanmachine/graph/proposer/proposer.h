/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "beanmachine/graph/graph.h"

// TODO: should change name to "Proposal" to better match the literature.

namespace beanmachine {
namespace proposer {

class Proposer {
 public:
  Proposer() {}
  /*
  Sample a value from the proposer.
  :param gen: Random number generator.
  :returns: A value.
  */
  virtual graph::NodeValue sample(std::mt19937& gen) const = 0;
  /*
  Compute the log_prob of a value.
  :param value: The value to evaluate the distribution.
  :returns: log probability of value.
  */
  virtual double log_prob(graph::NodeValue& value) const = 0;
  // Destructor for Proposer
  virtual ~Proposer() {}
};

/*
Return a unique pointer to a Proposer object.
:param value: The current value.
:param grad1: First gradient.
:param grad2: Second gradient.
:returns: A proposer object pointer.
*/
std::unique_ptr<Proposer>
nmc_proposer(const graph::NodeValue& value, double grad1, double grad2);

} // namespace proposer
} // namespace beanmachine
