// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include "beanmachine/graph/distribution.h"

namespace beanmachine {
namespace distribution {

class Beta : public Distribution {
 public:
  Beta(
      graph::AtomicType sample_type,
      const std::vector<graph::Node*>& in_nodes);
  ~Beta() override {}
  graph::AtomicValue sample(std::mt19937& gen) const override;
  double log_prob(const graph::AtomicValue& value) const override;
};

} // namespace distribution
} // namespace beanmachine
