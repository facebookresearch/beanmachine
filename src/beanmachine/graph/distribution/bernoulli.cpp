// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include "beanmachine/graph/distribution/bernoulli.h"

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
    throw std::invalid_argument("Bernoulli parent must be a probability");
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

// Note: Bernoulli log_prob is defined as  x log(p) + (1-x) log(1-p)
// where x is the value and p is the probability parameter
// grad w.r.t. x is  log(p) - log(1-p) ; grad2 = 0
// grad w.r.t p is   (x/p) * p' - ((1-x)/(1-p)) * p'
// grad2 w.r.t. p is -(x/p^2) * p'^2 + (x/p) * p'' - ((1-x)/(1-p)^2) * p'^2 -
// ((1-x)/(1-p)) * p''
void Bernoulli::gradient_log_prob_value(
    const graph::AtomicValue& /* value */,
    double& grad1,
    double& /* grad2 */) const {
  double prob = in_nodes[0]->value._double;
  grad1 += std::log(prob) - std::log(1 - prob);
  // grad2 += 0
}

void Bernoulli::gradient_log_prob_param(
    const graph::AtomicValue& value,
    double& grad1,
    double& grad2) const {
  double val = value._bool ? 1.0 : 0.0;
  double prob = in_nodes[0]->value._double;
  double prob1 = in_nodes[0]->grad1;
  double prob2 = in_nodes[0]->grad2;
  grad1 += ((val / prob) - ((1 - val) / (1 - prob))) * prob1;
  grad2 += ((val / prob) - ((1 - val) / (1 - prob))) * prob2 +
      (-(val / (prob * prob)) - ((1 - val) / ((1 - prob) * (1 - prob)))) *
          prob1 * prob1;
}

} // namespace distribution
} // namespace beanmachine
