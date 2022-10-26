/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/fluid_factory.h"
#include <memory>
#include <stdexcept>
#include "beanmachine/minibmg/distribution/normal.h"

namespace beanmachine::minibmg {

FluidDistribution::FluidDistribution(DistributionNode2p node) : node{node} {}

const FluidDistribution half_normal(Value2 stddev) {
  DistributionNode2p node =
      std::make_shared<const DistributionHalfNormalNode2>(stddev.node);
  return node;
}

const FluidDistribution normal(Value2 mean, Value2 stddev) {
  DistributionNode2p node =
      std::make_shared<const DistributionNormalNode2>(mean.node, stddev.node);
  return node;
}

const FluidDistribution beta(Value2 a, Value2 b) {
  DistributionNode2p node =
      std::make_shared<const DistributionBetaNode2>(a.node, b.node);
  return node;
}

const FluidDistribution bernoulli(Value2 p) {
  DistributionNode2p node =
      std::make_shared<const DistributionBernoulliNode2>(p.node);
  return node;
}

Value2 sample(const FluidDistribution& d, std::string rvid) {
  ScalarNode2p node = std::make_shared<const ScalarSampleNode2>(d.node, rvid);
  return node;
}

void Graph2::FluidFactory::observe(const Value2& sample, double value) {
  auto node = sample.node;
  if (!dynamic_cast<const ScalarSampleNode2*>(node.get())) {
    throw std::invalid_argument("can only observe a sample");
  }
  for (const auto& n : observations) {
    if (n.first == node) {
      throw std::invalid_argument("sample already observed");
    }
  }
  observations.push_back({node, value});
}

unsigned Graph2::FluidFactory::query(const Value2& value) {
  auto result = queries.size();
  queries.push_back(value.node);
  return result;
}

Graph2 Graph2::FluidFactory::build() {
  return Graph2::create(queries, observations);
}

} // namespace beanmachine::minibmg
