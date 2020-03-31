// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>
#include <random>

#include "beanmachine/graph/proposer/normal.h"

namespace beanmachine {
namespace proposer {

const double PI = 3.141592653589793;

graph::AtomicValue Normal::sample(std::mt19937& gen) const {
  std::normal_distribution<double> dist(mu, sigma);
  return graph::AtomicValue(graph::AtomicType::REAL, dist(gen));
}

double Normal::log_prob(graph::AtomicValue& value) const {
  return -std::log(sigma) - 0.5 * std::log(2 * PI)
    - 0.5 * (value._double - mu) * (value._double - mu) / (sigma * sigma);
}

} // namespace proposer
} // namespace beanmachine
