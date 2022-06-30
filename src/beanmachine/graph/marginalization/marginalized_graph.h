/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <functional>
#include "beanmachine/graph/distribution/dummy_marginal.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/marginalization/subgraph.h"

namespace beanmachine {
namespace graph {

void marginalize_graph(Graph& g, uint discrete_sample_node_id);

} // namespace graph
} // namespace beanmachine
