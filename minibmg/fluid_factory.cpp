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

FluidDistribution::FluidDistribution(DistributionNodep node) : node{node} {}

const FluidDistribution half_normal(Value stddev) {
  DistributionNodep node =
      std::make_shared<const DistributionHalfNormalNode>(stddev.node);
  return node;
}

const FluidDistribution normal(Value mean, Value stddev) {
  DistributionNodep node =
      std::make_shared<const DistributionNormalNode>(mean.node, stddev.node);
  return node;
}

const FluidDistribution beta(Value a, Value b) {
  DistributionNodep node =
      std::make_shared<const DistributionBetaNode>(a.node, b.node);
  return node;
}

const FluidDistribution bernoulli(Value p) {
  DistributionNodep node =
      std::make_shared<const DistributionBernoulliNode>(p.node);
  return node;
}

const FluidDistribution exponential(Value rate) {
  DistributionNodep node =
      std::make_shared<const DistributionExponentialNode>(rate.node);
  return node;
}

Value sample(const FluidDistribution& d, std::string rvid) {
  ScalarNodep node = std::make_shared<const ScalarSampleNode>(d.node, rvid);
  return node;
}

void Graph::FluidFactory::observe(const Value& sample, double value) {
  auto node = sample.node;
  if (!dynamic_cast<const ScalarSampleNode*>(node.get())) {
    throw std::invalid_argument("can only observe a sample");
  }
  for (const auto& n : observations) {
    if (n.first == node) {
      throw std::invalid_argument("sample already observed");
    }
  }
  observations.push_back({node, value});
}

unsigned Graph::FluidFactory::query(const Value& value) {
  auto result = queries.size();
  queries.push_back(value.node);
  return result;
}

Graph Graph::FluidFactory::build() {
  return Graph::create(queries, observations);
}

} // namespace beanmachine::minibmg
