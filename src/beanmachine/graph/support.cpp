/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"
#include "beanmachine/graph/support.h"

namespace beanmachine {
namespace graph {

using namespace std;

Support Graph::compute_support() {
  return _compute_support_given_mutable_choice(false);
}

MutableSupport Graph::compute_mutable_support() {
  return _compute_support_given_mutable_choice(true);
}

Support Graph::_compute_support_given_mutable_choice(bool mutable_only) {
  // we will do a standard BFS except that we are doing a BFS
  // in the reverse direction of the graph edges
  std::set<uint> visited;
  std::list<uint> queue;
  Support support;
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
    auto& node = nodes[node_id];
    if (!mutable_only or node->is_mutable()) {
      support.insert(node_id);
    }
    for (const auto& parent : node->in_nodes) {
      queue.push_back(parent->index);
    }
  }
  return support;
}

AffectedNodes Graph::compute_affected_nodes(
    uint root_id,
    const OrderedNodeIDs& ordered_node_ids) {
  return _compute_affected_nodes(root_id, ordered_node_ids, true);
}

AffectedNodes Graph::compute_affected_nodes_except_self(
    uint root_id,
    const OrderedNodeIDs& ordered_node_ids) {
  return _compute_affected_nodes(root_id, ordered_node_ids, false);
}

AffectedNodes Graph::_compute_affected_nodes(
    uint root_id,
    const OrderedNodeIDs& ordered_node_ids,
    bool include_root_node) {
  auto include = [&](uint node_id) {
    return ordered_node_ids.find(node_id) != ordered_node_ids.end() and
        (include_root_node or (node_id != root_id));
  };
  return _compute_affected_nodes(root_id, include);
}

AffectedNodes Graph::_compute_affected_nodes(
    uint root_id,
    function<bool(uint node_id)> include) {
  // check for the validity of root_id
  if (root_id >= nodes.size()) {
    throw std::out_of_range(
        "node_id (" + std::to_string(root_id) + ") must be less than " +
        std::to_string(nodes.size()));
  }
  DeterministicAffectedNodes det_affected_nodes;
  StochasticAffectedNodes sto_affected_nodes;
  // we will do a BFS starting from the current node
  // and ending at stochastic nodes
  std::set<uint> visited;
  std::list<uint> queue({root_id});
  while (not queue.empty()) {
    uint node_id = queue.front();
    queue.pop_front();
    if (visited.contains(node_id)) {
      continue;
    }
    visited.insert(node_id);

    auto& node = nodes[node_id];
    bool traverse_out_nodes = true;
    if (include(node_id)) {
      if (node->is_stochastic()) {
        sto_affected_nodes.push_back(node_id);
        if (node_id != root_id) { // if hit new stochastic node
          traverse_out_nodes = false; // don't go on
        }
      } else {
        det_affected_nodes.push_back(node_id);
      }
    }

    if (traverse_out_nodes) {
      for (const auto& out_node : node->out_nodes) {
        assert(out_node->index > node_id);
        queue.push_back(out_node->index);
      }
    }
  }

  // We must sort the node vectors to maintain topological order.
  // This may seem unnecessary since an out-node (child)
  // only gets visited after his in-node (parent) gets visited,
  // but we must remember that nodes may have multiple parents
  // and a second parent may actually have greater depth than its
  // child and end up being visited later.
  std::sort(det_affected_nodes.begin(), det_affected_nodes.end());
  std::sort(sto_affected_nodes.begin(), sto_affected_nodes.end());

  return {det_affected_nodes, sto_affected_nodes};
}

std::tuple<DeterministicAncestors, StochasticAncestors>
collect_deterministic_and_stochastic_ancestors(Graph& graph) {
  DeterministicAncestors det_anc(graph.node_ptrs().size());
  StochasticAncestors sto_anc(graph.node_ptrs().size());
  for (Node* node : graph.node_ptrs()) {
    OrderedNodeIDs det_set;
    OrderedNodeIDs sto_set;
    for (Node* parent : node->in_nodes) {
      if (parent->is_stochastic()) {
        sto_set.insert(parent->index);
      } else {
        auto parent_det_anc = det_anc[parent->index]; // NOLINT
        det_set.insert(parent_det_anc.begin(), parent_det_anc.end());
        if (parent->node_type == NodeType::OPERATOR) {
          det_set.insert(parent->index);
        }
        auto parent_sto_anc = sto_anc[parent->index]; // NOLINT
        sto_set.insert(parent_sto_anc.begin(), parent_sto_anc.end());
      }
    }
    std::vector<uint>& node_det_anc = det_anc[node->index]; // NOLINT
    std::vector<uint>& node_sto_anc = sto_anc[node->index]; // NOLINT
    node_det_anc.insert(node_det_anc.end(), det_set.begin(), det_set.end());
    node_sto_anc.insert(node_sto_anc.end(), sto_set.begin(), sto_set.end());
  }
  return {det_anc, sto_anc};
}

} // namespace graph
} // namespace beanmachine
