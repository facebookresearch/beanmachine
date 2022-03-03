/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/marginalization/subgraph.h"

namespace beanmachine {
namespace graph {

SubGraph::SubGraph(Graph& g) : graph(g) {}

void SubGraph::add_node_by_id(uint node_id) {
  pending_node_ids.insert(node_id);
}

void SubGraph::move_nodes_from_graph() {
  // indices are from largest to smallest
  for (std::set<uint>::reverse_iterator rit = pending_node_ids.rbegin();
       rit != pending_node_ids.rend();
       rit++) {
    uint index = *rit;
    nodes.push_back(std::move(graph.nodes[index]));
    graph.nodes.erase(graph.nodes.begin() + index);
  }
  // reorder nodes from smallest index to largest
  std::reverse(nodes.begin(), nodes.end());
  graph.reindex_nodes();
  reindex_nodes();
}
} // namespace graph
} // namespace beanmachine
