/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/marginalization/subgraph.h"
#include <algorithm>
#include <memory>
#include "beanmachine/graph/marginalization/copy_node.h"

namespace beanmachine {
namespace graph {

SubGraph::SubGraph(Graph& g) : graph(g) {}

void SubGraph::add_node_by_id(uint node_id) {
  pending_node_ids.insert(node_id);
}

bool SubGraph::has_node(uint node_id) {
  return std::find(pending_node_ids.begin(), pending_node_ids.end(), node_id) !=
      pending_node_ids.end();
}

void SubGraph::link_copy_node(Node* node, Node* copy_node) {
  copy_map[copy_node] = node;
}

void SubGraph::move_nodes_from_graph(
    uint insertion_index,
    uint insertion_size) {
  /*
  SubGraph has a list of "pending node ids" which should be
  moved from the graph to the subgraph

  The graph `nodes` vector was originally
  index 0: 0
  index 1: 1
  ...
  index n: n

  After after the marginalized nodes were added
  the graph `nodes` vector now looks like:
  index 0: 0
  index 1: 1
  ...
  index insertion_index: ?
  index (insertion_index + 1): ?
  ...
  index (insertion_index + insertion_size - 1): ?
  index (insertion_index + insertion_size): insertion_index
  index (insertion_index + insertion_size + 1): insertion_index + 1
  ...
  index (n + insertion_size): n

  We want to move the pending node ids corresponding to the
  original graph ids in the list, even though parts of the
  graph are now shifted by `insertion_index`
  */

  // copy nodes of parents should be at the beginning of the nodes
  uint copy_nodes_size = nodes.size();
  // indices are from largest to smallest
  for (std::set<uint>::reverse_iterator rit = pending_node_ids.rbegin();
       rit != pending_node_ids.rend();
       rit++) {
    uint index = *rit;
    // index in graph.nodes has increased because it is after
    // the marginalized_node insertion
    if (index >= insertion_index) {
      index += insertion_size;
    }
    // reorder nodes after copy nodes from smallest index to largest
    nodes.insert(
        nodes.begin() + copy_nodes_size, std::move(graph.nodes[index]));
    graph.nodes.erase(graph.nodes.begin() + index);
  }
  graph.reindex_nodes();
  reindex_nodes();
}
} // namespace graph
} // namespace beanmachine
