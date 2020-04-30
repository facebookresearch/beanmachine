// Copyright (c) Facebook, Inc. and its affiliates.
#include <string>

#include "beanmachine/graph/distribution/flat.h"

namespace beanmachine {
namespace distribution {

using namespace graph;

Flat::Flat(AtomicType sample_type, const std::vector<Node*>& in_nodes)
    : Distribution(DistributionType::FLAT, sample_type) {
  // a Flat distribution has no parents
  if (in_nodes.size() != 0) {
    throw std::invalid_argument("Flat distribution has no parents");
  }
  // almost all types are supported
  if (sample_type == AtomicType::TENSOR) {
    throw std::invalid_argument("Flat doesn't support Tensor samples");
  }
}

AtomicValue Flat::sample(std::mt19937& gen) const {
  AtomicValue value;
  switch (sample_type) {
    case AtomicType::BOOLEAN: {
      std::bernoulli_distribution dist(0.5);
      value = AtomicValue(dist(gen));
      break;
    }
    case AtomicType::PROBABILITY: {
      std::uniform_real_distribution<double> dist(0, 1);
      value = AtomicValue(AtomicType::PROBABILITY, dist(gen));
      break;
    }
    case AtomicType::REAL: {
      std::uniform_real_distribution<double> dist(
          std::numeric_limits<double>::lowest(),
          std::numeric_limits<double>::max());
      value = AtomicValue(dist(gen));
      break;
    }
    case AtomicType::POS_REAL: {
      std::uniform_real_distribution<double> dist(
          0, std::numeric_limits<double>::max());
      value = AtomicValue(AtomicType::POS_REAL, dist(gen));
      break;
    }
    case AtomicType::NATURAL: {
      std::uniform_int_distribution<natural_t> dist(
          0, std::numeric_limits<natural_t>::max());
      value = AtomicValue((natural_t)dist(gen));
      break;
    }
    default: {
      assert(false);
    }
  }
  return value;
}

// A Flat distribution is really easy in terms of computing the log_prob and the
// gradients of the log_prob. These are all zero!

double Flat::log_prob(const AtomicValue& /* value */) const {
  return 0;
}

void Flat::gradient_log_prob_value(
    const AtomicValue& /* value */,
    double& /* grad1 */,
    double& /* grad2 */) const {}

void Flat::gradient_log_prob_param(
    const AtomicValue& /* value */,
    double& /* grad1 */,
    double& /* grad2 */) const {}

} // namespace distribution
} // namespace beanmachine
