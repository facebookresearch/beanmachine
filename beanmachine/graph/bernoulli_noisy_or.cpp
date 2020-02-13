// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include "beanmachine/graph/bernoulli_noisy_or.h"

namespace beanmachine {
namespace distribution {

BernoulliNoisyOr::BernoulliNoisyOr(
    graph::AtomicType sample_type,
    const std::vector<graph::Node*>& in_nodes)
    : Distribution(graph::DistributionType::BERNOULLI, sample_type) {
  if (sample_type != graph::AtomicType::BOOLEAN) {
    throw std::invalid_argument(
      "BernoulliNoisyOr produces boolean valued samples");
  }
  // a Bernoulli can only have one parent which must look like a probability
  if (in_nodes.size() != 1) {
    throw std::invalid_argument(
        "BernoulliNoisyOr distribution must have exactly one parent");
  }
  // if the parent is a constant then we can directly check the type
  if (in_nodes[0]->node_type == graph::NodeType::CONSTANT) {
    graph::AtomicValue constant = in_nodes[0]->value;
    if (constant.type != graph::AtomicType::REAL) {
      throw std::invalid_argument(
          "BernoulliNoisyOr parent probability must be real-valued");
    }
    // all probabilities must be greater than 0
    if (constant._double < 0) {
      throw std::invalid_argument(
          "BernoulliNoisyOr parameter must be >= 0");
    }
  }
}

graph::AtomicValue BernoulliNoisyOr::sample(std::mt19937& gen) const {
  double param = in_nodes[0]->value._double;
  double prob = 1 - exp(-param);
  std::bernoulli_distribution distrib(prob);
  return graph::AtomicValue((bool)distrib(gen));
}

double BernoulliNoisyOr::log_prob(const graph::AtomicValue& value) const {
  double param = in_nodes[0]->value._double;
  if (not value._bool) {
    return -param;
  }
  // see the following document for an explanation of why we switch at .69315
  // https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
  // essentially for small values expm1 prevents underflow to 0.0 and for
  // large values it prevents overflow to 1.0.
  //   log1mexp(1e-20) -> -46.051701859880914
  //   log1mexp(40) -> -4.248354255291589e-18
  if (param < 0.69315) {
    return std::log(-std::expm1(-param));
  }
  else {
    return std::log1p(-std::exp(-param));
  }
}

} // namespace distribution
} // namespace beanmachine
