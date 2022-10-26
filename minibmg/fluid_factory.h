/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "beanmachine/minibmg/ad/traced.h"
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/node.h"

namespace beanmachine::minibmg {

// For values
using Value = Traced;

// For distributions
class FluidDistribution {
 public:
  DistributionNodep node;
  /* implicit */ FluidDistribution(DistributionNodep node);
};

const FluidDistribution half_normal(Value stddev);
const FluidDistribution normal(Value mean, Value stddev);
const FluidDistribution beta(Value a, Value b);
const FluidDistribution bernoulli(Value p);
const FluidDistribution exponential(Value rate);

Value sample(const FluidDistribution& d, std::string rvid = make_fresh_rvid());

class Graph::FluidFactory {
 public:
  void observe(const Traced& sample, double value);
  unsigned query(const Traced& value);
  Graph build();

 private:
  std::vector<Nodep> queries;
  std::list<std::pair<Nodep, double>> observations;
};

} // namespace beanmachine::minibmg
