// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include "beanmachine/graph/bernoulli.h"

namespace beanmachine {
namespace distribution {

Bernoulli::Bernoulli(
    graph::AtomicType sample_type,
    const std::vector<graph::Node*>& in_nodes)
    : Distribution(graph::DistributionType::BERNOULLI, sample_type) {
  if (sample_type != graph::AtomicType::BOOLEAN) {
    throw std::invalid_argument("Bernoulli produces boolean valued samples");
  }
  // a Bernoulli can only have one parent which must be a probability
  if (in_nodes.size() != 1) {
    throw std::invalid_argument(
        "Bernoulli distribution must have exactly one parent");
  }
  // check the parent type
  const auto& parent0 = in_nodes[0]->value;
  if (parent0.type != graph::AtomicType::PROBABILITY) {
    throw std::invalid_argument(
        "Bernoulli parent must be a probability");
  }
}

graph::AtomicValue Bernoulli::sample(std::mt19937& gen) const {
  std::bernoulli_distribution distrib(in_nodes[0]->value._double);
  return graph::AtomicValue((bool)distrib(gen));
}

double Bernoulli::log_prob(const graph::AtomicValue& value) const {
  double prob = in_nodes[0]->value._double;
  return value._bool ? std::log(prob) : std::log(1 - prob);
}

} // namespace distribution
} // namespace beanmachine
