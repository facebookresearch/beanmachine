/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/marginalization/subgraph.h"
#include <algorithm>
#include <cstddef>
#include <memory>

namespace beanmachine {
namespace graph {

SubGraph::SubGraph(Graph& g) : graph(g) {}

std::set<uint> SubGraph::get_node_ids() {
  return pending_node_ids;
}

void SubGraph::add_node_by_id(uint node_id) {
  pending_node_ids.insert(node_id);
}

bool SubGraph::has_node(uint node_id) {
  return std::find(pending_node_ids.begin(), pending_node_ids.end(), node_id) !=
      pending_node_ids.end();
}

void SubGraph::move_nodes_from_graph_and_reindex() {
  /*
  SubGraph has a list of "pending node ids" which should be
  moved from the graph to the subgraph
  */

  // copy nodes of parents should be at the beginning of the nodes
  uint parent_nodes_size = static_cast<uint>(nodes.size());
  // indices are from largest to smallest
  for (std::set<uint>::reverse_iterator rit = pending_node_ids.rbegin();
       rit != pending_node_ids.rend();
       rit++) {
    uint index = *rit;
    // insert nodes after parent nodes from smallest index to largest
    nodes.insert(
        nodes.begin() + parent_nodes_size, std::move(graph.nodes[index]));
    // TODO: replace erase with assigning list of new_nodes to graph.nodes
    graph.nodes.erase(graph.nodes.begin() + index);
  }
  graph.reindex_nodes();
  reindex_nodes();
}
} // namespace graph
} // namespace beanmachine
