// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include "beanmachine/graph/beta.h"

namespace beanmachine {
namespace distribution {

Beta::Beta(
    graph::AtomicType sample_type,
    const std::vector<graph::Node*>& in_nodes)
    : Distribution(graph::DistributionType::BETA, sample_type) {
  // a Beta has two parents which are real numbers and it outputs a probability
  if (sample_type != graph::AtomicType::PROBABILITY) {
    throw std::invalid_argument("Beta produces probability samples");
  }
  if (in_nodes.size() != 2) {
    throw std::invalid_argument(
        "Beta distribution must have exactly two parents");
  }
  if (in_nodes[0]->value.type != graph::AtomicType::REAL
      or in_nodes[1]->value.type != graph::AtomicType::REAL) {
    throw std::invalid_argument("Beta parents must be real-valued");
  }
}

graph::AtomicValue Beta::sample(std::mt19937& gen) const {
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  std::gamma_distribution<double> distrib_a(param_a, 1);
  std::gamma_distribution<double> distrib_b(param_b, 1);
  double x = distrib_a(gen);
  double y = distrib_b(gen);
  return graph::AtomicValue(graph::AtomicType::PROBABILITY, x / (x + y));
}

double Beta::log_prob(const graph::AtomicValue& value) const {
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  double ret_val = (param_a - 1) * log(value._double)
    + (param_b - 1) * log(1 - value._double);
  ret_val += lgamma(param_a + param_b) - lgamma(param_a) - lgamma(param_b);
  return ret_val;
}

} // namespace distribution
} // namespace beanmachine
