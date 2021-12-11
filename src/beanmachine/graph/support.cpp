/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"

namespace beanmachine {
namespace graph {

// the support of a graph is the set of operator and factor nodes that are
// needed to determine the value of query and observed variables.
// In other words, it is the set of queried and observed variables themselves
// plus their ancestors that are operator and factor nodes.
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

void include_children(const Node* node, std::list<uint>& queue) {
  for (const auto& child : node->out_nodes) {
    queue.push_back(child->index);
  }
}

std::tuple<std::vector<uint>, std::vector<uint>> Graph::compute_affected_nodes(
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
    bool include_children_of_node;
    if (node->is_stochastic()) {
      if (support.find(node_id) != support.end()) {
        // no need to check if node is operator because
        // all stochastic nodes are operators
        sto_desc.push_back(node_id);
      }
      // we only proceed to include children if node is root;
      // otherwise we stop because nodes beyond
      // non-root stochastic nodes are not directly
      // affected by changes of value in the root.
      include_children_of_node = (node_id == root_id);
    } else if (
        node->node_type == NodeType::OPERATOR and
        support.find(node_id) != support.end()) {
      det_desc.push_back(node_id);
      // We always include children of deterministic
      // nodes because the definition of affected nodes
      // includes all deterministic nodes up to
      // the first encountered stochastic nodes.
      include_children_of_node = true;
    } else {
      // We include children of other types of
      // nodes because we must go on until we
      // find the first stochastic descendants.
      include_children_of_node = true;
    }

    if (include_children_of_node) {
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
