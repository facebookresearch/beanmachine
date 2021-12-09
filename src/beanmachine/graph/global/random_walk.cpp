/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/random_walk.h"
#include "beanmachine/graph/global/proposer/random_walk_proposer.h"

namespace beanmachine {
namespace graph {

RandomWalkMH::RandomWalkMH(Graph& g, double step_size) : GlobalMH(g) {
  proposer =
      std::make_unique<RandomWalkProposer>(RandomWalkProposer(step_size));
}

} // namespace graph
} // namespace beanmachine
