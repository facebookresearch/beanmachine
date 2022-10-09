/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <beanmachine/graph/graph.h>

namespace beanmachine {
namespace graph {

using DeterministicAncestors = std::vector<std::vector<NodeID>>;
using StochasticAncestors = std::vector<std::vector<NodeID>>;

/*
Returns two vectors of vectors of node ids.
The first vector contains, for each node in topological order,
the indices of deterministic ancestors of that node.
The second vector analogously contains stochastic ancestors.
*/
std::tuple<DeterministicAncestors, StochasticAncestors>
collect_deterministic_and_stochastic_ancestors(Graph& graph);

} // namespace graph
} // namespace beanmachine
