// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/graph.h"

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
  virtual graph::AtomicValue sample(std::mt19937& gen) const = 0;
  /*
  Compute the log_prob of a value.
  :param value: The value to evaluate the distribution.
  :returns: log probability of value.
  */
  virtual double log_prob(graph::AtomicValue& value) const = 0;
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
nmc_proposer(const graph::AtomicValue& value, double grad1, double grad2);

/*
Returns a value for the specified type uniformly at random.
:param gen: A random number generator
:param type: The desired type
:returns: A value of this type
*/
graph::AtomicValue uniform_initializer(
    std::mt19937& gen,
    graph::ValueType type);

} // namespace proposer
} // namespace beanmachine
