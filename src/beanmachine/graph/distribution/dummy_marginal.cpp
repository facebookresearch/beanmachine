/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include "beanmachine/graph/distribution/dummy_marginal.h"
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace distribution {

DummyMarginal::DummyMarginal(std::unique_ptr<graph::SubGraph> subgraph_ptr)
    : Distribution(graph::DistributionType::DUMMY, graph::AtomicType::REAL) {
  this->subgraph_ptr = std::move(subgraph_ptr);
}

} // namespace distribution
} // namespace beanmachine
