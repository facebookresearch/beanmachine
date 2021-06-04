// Copyright (c) Facebook, Inc. and its affiliates.
#include <algorithm>

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"

namespace beanmachine {
namespace graph {

// the support of a graph is the set of operator and factor nodes that are the
// ancestors of the queried and observed variables
std::set<uint> Graph::compute_support() {
  // we will do a standard BFS except that we are doing a BFS
  // in the reverse direction of the graph edges
  std::set<uint> visited;
  std::list<uint> queue;
  std::set<uint> support;
  // initialize BFS queue with all the observed and queried nodes since the
  // parents of these nodes define the support of the graph
  for (uint node_id : observed) {
    queue.push_back(node_id);
  }
  for (uint node_id : queries) {
    queue.push_back(node_id);
  }
  // BFS loop
  while (not queue.empty()) {
    uint node_id = queue.front();
    queue.pop_front();
    if (visited.find(node_id) != visited.end()) {
      continue;
    }
    visited.insert(node_id);
    const Node* node = nodes[node_id].get();
    if (node->node_type == NodeType::OPERATOR or
        node->node_type == NodeType::FACTOR) {
      support.insert(node_id);
    }
    for (const auto& parent : node->in_nodes) {
      queue.push_back(parent->index);
    }
  }
  return support;
}

private
void include_children(const Node* node, std::list<uint>& queue) {
  for (const auto& child : node->out_nodes) {
    queue.push_back(child->index);
  }
}

std::tuple<std::vector<uint>, std::vector<uint>>
Graph::get_nodes_up_to_immediate_stochastic_descendants(
    uint root_id,
    const std::set<uint>& support) {
  // check for the validity of root_id since this method is not private
  if (root_id >= nodes.size()) {
    throw std::out_of_range(
        "node_id (" + std::to_string(root_id) + ") must be less than " +
        std::to_string(nodes.size()));
  }
  std::vector<uint> det_desc;
  std::vector<uint> sto_desc;
  // we will do a BFS starting from the current node and ending at stochastic
  // nodes
  std::set<uint> visited;
  std::list<uint> queue({root_id});
  // BFS loop
  while (not queue.empty()) {
    uint node_id = queue.front();
    queue.pop_front();
    if (visited.find(node_id) != visited.end()) {
      continue;
    }
    visited.insert(node_id);
    const Node* node = nodes[node_id].get();
    bool include_node_s_children;
    if (node->is_stochastic()) {
      if (support.find(node_id) != support.end()) {
        // no need to check if node is operator because
        // all stochastic nodes are operators
        sto_desc.push_back(node_id);
      }
      include_node_s_children = (node_id == root_id);
    } else if (
        node->node_type == NodeType::OPERATOR and
        support.find(node_id) != support.end()) {
      det_desc.push_back(node_id);
      include_node_s_children = true;
    } else {
      include_node_s_children = true;
    }

    if (include_node_s_children) {
      include_children(node, queue);
    }
  }
  std::sort(det_desc.begin(), det_desc.end());
  std::sort(sto_desc.begin(), sto_desc.end());
  return std::make_tuple(det_desc, sto_desc);
}

// compute the ancestors of the current node
// returns vector of deterministic nodes and vector of stochastic nodes
// that are operators and ancestors of the current node
// NOTE: we don't return ancestors of stochastic ancestors
// NOTE: current node is not returned
std::tuple<std::vector<uint>, std::vector<uint>> Graph::compute_ancestors(
    uint root_id) {
  // check for the validity of root_id since this method is not private
  if (root_id >= nodes.size()) {
    throw std::out_of_range(
        "node_id (" + std::to_string(root_id) + ") must be less than " +
        std::to_string(nodes.size()));
  }
  const Node* root = nodes[root_id].get();
  return std::make_tuple(root->det_anc, root->sto_anc);
}

} // namespace graph
} // namespace beanmachine
