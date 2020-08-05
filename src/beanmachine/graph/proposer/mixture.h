// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/proposer/proposer.h"

namespace beanmachine {
namespace proposer {

class Mixture : public Proposer {
 public:
  /*
  Constructor for Mixture
  :param probabilities:
  :param proposers:
  */
  Mixture(
      std::vector<double> in_weights,
      std::vector<std::unique_ptr<Proposer>> in_proposers)
      : Proposer(), weights(in_weights), proposers(std::move(in_proposers)) {
    assert(weights.size() == proposers.size());
    weight_sum = 0;
    for (auto weight : weights) {
      weight_sum += weight;
    }
  }
  /*
  Sample a value from the proposer.
  :param gen: Random number generator.
  :returns: A value.
  */
  graph::AtomicValue sample(std::mt19937& gen) const override;
  /*
  Compute the log_prob of a value.
  :param value: The value to evaluate the distribution.
  :returns: log probability of value.
  */
  double log_prob(graph::AtomicValue& value) const override;

 private:
  double weight_sum;
  std::vector<double> weights;
  std::vector<std::unique_ptr<Proposer>> proposers;
};

} // namespace proposer
} // namespace beanmachine
