// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <string>
#include "graph.h"

namespace beanmachine {
namespace distribution {

class Distribution : public graph::Node {
 public:
  static std::unique_ptr<Distribution> new_distribution(
      graph::DistributionType dist_type,
      graph::AtomicType sample_type,
      const std::vector<graph::Node*>& in_nodes);

  Distribution(graph::DistributionType dist_type, graph::AtomicType sample_type)
      : graph::Node(graph::NodeType::DISTRIBUTION),
        dist_type(dist_type),
        sample_type(sample_type) {}
  virtual graph::AtomicValue sample(std::mt19937& gen) const = 0;
  void eval(std::mt19937& /* */) override {
    throw std::runtime_error(
        "internal error: eval() is not implemented for distribution");
  }
  virtual double log_prob(const graph::AtomicValue& value) const = 0;
  graph::DistributionType dist_type;
  graph::AtomicType sample_type;
};

} // namespace distribution
} // namespace beanmachine
