/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "beanmachine/minibmg/ad/traced2.h"
#include "beanmachine/minibmg/graph2.h"
#include "beanmachine/minibmg/node2.h"

namespace beanmachine::minibmg {

// For values
using Value2 = Traced2;

// For distributions
class FluidDistribution {
 public:
  DistributionNode2p node;
  /* implicit */ FluidDistribution(DistributionNode2p node);
};

const FluidDistribution half_normal(Value2 stddev);
const FluidDistribution normal(Value2 mean, Value2 stddev);
const FluidDistribution beta(Value2 a, Value2 b);
const FluidDistribution bernoulli(Value2 p);

Value2 sample(const FluidDistribution& d, std::string rvid = make_fresh_rvid());

class Graph2::FluidFactory {
 public:
  void observe(const Traced2& sample, double value);
  unsigned query(const Traced2& value);
  Graph2 build();

 private:
  std::vector<Node2p> queries;
  std::list<std::pair<Node2p, double>> observations;
};

} // namespace beanmachine::minibmg
