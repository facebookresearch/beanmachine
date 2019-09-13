// Copyright (c) Facebook, Inc. and its affiliates.
#include <beanmachine/graph/graph.h>
#include <beanmachine/graph/operator.h>

namespace beanmachine {
namespace graph {

// the support of a graph is the set of nodes that are the ancestors of the
// queried and observed variables
std::set<uint> Graph::compute_support() {
  // we will do a standard BFS except that we are doing a BFS
  // in the reverse direction of the graph edges
  std::set<uint> visited;
  std::list<uint> queue;
  // initialize BFS queue with all the observed and queried nodes since the
  // parents of these nodes define the support of the graph
  for (uint node_id : observed) {
    queue.push_back(node_id);
  }
  for (uint node_id : queries) {
    queue.push_back(node_id);
  }
  // BFS loop
  while (queue.size() > 0) {
    uint node_id = queue.front();
    queue.pop_front();
    if (visited.find(node_id) != visited.end()) {
      continue;
    }
    visited.insert(node_id);
    const Node* node = nodes[node_id].get();
    for (const auto& parent : node->in_nodes) {
      queue.push_back(parent->index);
    }
  }
  return visited;
}

// descendants are the inverse of support. Returns the deterministic and
// stochastic descendants each in their topological order. Note we compute
// the minimal set of descendants whose value or probablity needs to be changed.
// NOTE: the current node will also be returned.
std::tuple<std::list<uint>, std::list<uint>> Graph::compute_descendants(
    uint root_id) {
  // check for the validity of root_id since this method is not private
  if (root_id >= nodes.size()) {
    throw std::out_of_range(
      "node_id (" + std::to_string(root_id)
      + ") must be less than " + std::to_string(nodes.size()));
  }
  std::list<uint> det_desc;
  std::list<uint> sto_desc;
  // we will do a BFS starting from the current node and ending at stochastic
  // nodes
  std::set<uint> visited;
  std::list<uint> queue({root_id});
  // BFS loop
  while (queue.size() > 0) {
    uint node_id = queue.front();
    queue.pop_front();
    if (visited.find(node_id) != visited.end()) {
      continue;
    }
    visited.insert(node_id);
    const Node* node = nodes[node_id].get();
    // we stop looking at descendants when we hit a stochastic node
    // other than the root of this subgraph
    if (node->is_stochastic()) {
      sto_desc.push_back(node_id);
      if (node_id != root_id) {
        continue;
      }
    } else {
      det_desc.push_back(node_id);
    }
    for (const auto& child : node->out_nodes) {
      queue.push_back(child->index);
    }
  }
  det_desc.sort();
  sto_desc.sort();
  return std::make_tuple(det_desc, sto_desc);
}

} // namespace graph
} // namespace beanmachine
