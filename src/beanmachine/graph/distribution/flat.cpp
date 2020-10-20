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
}

bool Flat::_bool_sampler(std::mt19937& gen) const {
  std::bernoulli_distribution dist(0.5);
  return (bool)dist(gen);
}

double Flat::_double_sampler(std::mt19937& gen) const {
  std::uniform_real_distribution<double> dist;
  switch (sample_type.atomic_type) {
    case graph::AtomicType::REAL:
      dist = std::uniform_real_distribution<double>(
          std::numeric_limits<double>::lowest(),
          std::numeric_limits<double>::max());
      break;
    case graph::AtomicType::POS_REAL:
      dist = std::uniform_real_distribution<double>(
          0, std::numeric_limits<double>::max());
      break;
    case graph::AtomicType::PROBABILITY:
      dist = std::uniform_real_distribution<double>(0, 1);
      break;
    default:
      throw std::runtime_error(
          "Unsupported sample type for _double_sampler of Flat.");
  }
  return dist(gen);
}

natural_t Flat::_natural_sampler(std::mt19937& gen) const {
  std::uniform_int_distribution<natural_t> dist(
      0, std::numeric_limits<natural_t>::max());
  return (natural_t)dist(gen);
}

// A Flat distribution is really easy in terms of computing the log_prob and the
// gradients of the log_prob. These are all zero!

double Flat::log_prob(const NodeValue& /* value */) const {
  return 0;
}

void Flat::gradient_log_prob_value(
    const NodeValue& /* value */,
    double& /* grad1 */,
    double& /* grad2 */) const {}

void Flat::gradient_log_prob_param(
    const NodeValue& /* value */,
    double& /* grad1 */,
    double& /* grad2 */) const {}

} // namespace distribution
} // namespace beanmachine
