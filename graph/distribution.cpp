// Copyright (c) Facebook, Inc. and its affiliates.
#include <folly/String.h>

#include "bernoulli.h"
#include "distribution.h"
#include "tabular.h"

namespace beanmachine {
namespace distribution {

std::unique_ptr<Distribution> Distribution::new_distribution(
    graph::DistributionType dist_type,
    graph::AtomicType sample_type,
    const std::vector<graph::Node*>& in_nodes) {
  // check parent nodes are of the correct type
  for (graph::Node* parent : in_nodes) {
    if (parent->node_type == graph::NodeType::DISTRIBUTION) {
      throw std::invalid_argument("distribution parents can't be distribution");
    }
  }
  // now simply call the appropriate distribution constructor
  if (dist_type == graph::DistributionType::TABULAR) {
    return std::make_unique<Tabular>(sample_type, in_nodes);
  } else if (dist_type == graph::DistributionType::BERNOULLI) {
    return std::make_unique<Bernoulli>(sample_type, in_nodes);
  }
  throw std::invalid_argument(folly::stringPrintf(
      "Unknown distribution %d", static_cast<int>(dist_type)));
}

} // namespace distribution
} // namespace beanmachine
