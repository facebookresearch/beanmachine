// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/proposer/proposer.h"

namespace beanmachine {
namespace proposer {

class Delta : public Proposer {
 public:
  /*
  Delta puts all its mass at a single point
  :param alpha: point
  */
  Delta(graph::AtomicValue point) : Proposer(), point(point) {}
  /*
  Sample a value from the proposer.
  :param gen: Random number generator.
  :returns: A value.
  */
  graph::AtomicValue sample(std::mt19937& gen) const override {
    return point;
  }
  /*
  Compute the log_prob of a value.
  :param value: The value to evaluate the distribution.
  :returns: log probability of value.
  */
  double log_prob(graph::AtomicValue& value) const override {
    return value == point ? 0.0 : -std::numeric_limits<double>::infinity();
  };

 private:
  graph::AtomicValue point;
};

} // namespace proposer
} // namespace beanmachine
