/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/marginalization/marginalized_graph.h"
#include <algorithm>
#include "beanmachine/graph/distribution/dummy_marginal.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/marginalization/copy_node.h"
#include "beanmachine/graph/marginalization/subgraph.h"

namespace beanmachine {
namespace graph {

MarginalizedGraph::MarginalizedGraph(Graph& g) : Graph(g) {
  // iterate over discrete variables and rewrite graph
}

void MarginalizedGraph::marginalize(uint discrete_sample_node_id) {
  Node* discrete_sample = get_node(discrete_sample_node_id);
  Node* discrete_distribution = discrete_sample->in_nodes[0];

  std::unique_ptr<SubGraph> subgraph_ptr = std::make_unique<SubGraph>(*this);
  SubGraph* subgraph = subgraph_ptr.get();

  // compute nodes up to and including stochastic children of discrete_sample
  std::set<uint> ordered_support_node_ids =
      compute_full_ordered_support_node_ids();
  std::vector<uint> det_node_ids;
  std::vector<uint> sto_node_ids;
  std::tie(det_node_ids, sto_node_ids) =
      compute_children(discrete_sample->index, ordered_support_node_ids);

  // create MarginalDistribution
  std::unique_ptr<distribution::DummyMarginal>
      marginal_distribution_node_pointer =
          std::make_unique<distribution::DummyMarginal>(
              std::move(subgraph_ptr));
  marginal_distribution_node_pointer.get()->sample_type = AtomicType::REAL;
  distribution::DummyMarginal* marginal_distribution =
      marginal_distribution_node_pointer.get();
  nodes.push_back(std::move(marginal_distribution_node_pointer));

  // add nodes to subgraph
  // add discrete distribution and samples
  subgraph->add_node_by_id(discrete_distribution->index);
  subgraph->add_node_by_id(discrete_sample->index);
  // add all intermediate deterministic nodes to subgraph
  for (uint id : det_node_ids) {
    subgraph->add_node_by_id(id);
  }
  // add all stochastic nodes to subgraph
  for (uint id : sto_node_ids) {
    if (id != discrete_sample->index) {
      subgraph->add_node_by_id(id);
    }
  }

  // parents for MarginalDistribution
  // add parents of discrete_distribution
  for (Node* discrete_parent : discrete_distribution->in_nodes) {
    connect_parent_to_marginal_distribution(
        marginal_distribution, discrete_parent);
  }
  // add parents of deterministic children
  for (uint det_node_id : det_node_ids) {
    Node* det_node = get_node(det_node_id);
    for (Node* det_parent : det_node->in_nodes) {
      connect_parent_to_marginal_distribution(
          marginal_distribution, det_parent);
    }
  }
  // add parents of stochastic children
  for (uint sto_node_id : sto_node_ids) {
    Node* sto_node = get_node(sto_node_id);
    for (Node* sto_parent : sto_node->in_nodes) {
      connect_parent_to_marginal_distribution(
          marginal_distribution, sto_parent);
    }
  }

  // children for MarginalDistribution
  // add all children of discrete sample's stochastic children
  for (uint id : sto_node_ids) {
    // create "COPY" of child nodes as output of subgraph
    Node* node = get_node(id);
    std::unique_ptr<CopyNode> copy_node = std::make_unique<CopyNode>(node);
    subgraph->link_copy_node(node, copy_node.get());
    // only move children that are not in subgraph
    move_children_if(node, copy_node.get(), [&](Node* child) {
      return !subgraph->has_node(child->index);
    });
    // create link between marginal_distribution and copy_node
    marginal_distribution->out_nodes.push_back(copy_node.get());
    copy_node.get()->in_nodes.push_back(marginal_distribution);
    // add copy node to marginalized_graph
    nodes.push_back(std::move(copy_node));
  }

  // create "COPY" of parent nodes inside subgraph
  for (Node* parent : marginal_distribution->in_nodes) {
    std::unique_ptr<CopyNode> copy_node = std::make_unique<CopyNode>(parent);
    subgraph->link_copy_node(parent, copy_node.get());
    // only move children that are in subgraph to copy_node
    move_children_if(parent, copy_node.get(), [&](Node* child) {
      return subgraph->has_node(child->index);
    });
    subgraph->nodes.push_back(std::move(copy_node));
  }

  // move nodes to subgraph and finalize
  subgraph->move_nodes_from_graph();
}

void MarginalizedGraph::connect_parent_to_marginal_distribution(
    distribution::DummyMarginal* node,
    Node* parent) {
  // check that parent is not already in list of parents
  // check that parent is not part of subgraph
  if ((std::find(node->in_nodes.begin(), node->in_nodes.end(), parent) ==
       node->in_nodes.end()) and
      !node->subgraph_ptr.get()->has_node(parent->index)) {
    node->in_nodes.push_back(parent);
    parent->out_nodes.push_back(node);
  }
}

void MarginalizedGraph::move_children_if(
    Node* current_parent,
    Node* new_parent,
    std::function<bool(Node*)> condition) {
  // move children from current_parent to new_parent
  uint i = 0;
  while (i < current_parent->out_nodes.size()) {
    Node* child = current_parent->out_nodes[i];
    // only move children which meet condition
    if (condition(child)) {
      // remove child and current_parent connection
      current_parent->out_nodes.erase(current_parent->out_nodes.begin() + i);
      auto child_in_nodes_position = child->in_nodes.erase(std::find(
          child->in_nodes.begin(), child->in_nodes.end(), current_parent));
      // add child and new_parent connection
      new_parent->out_nodes.push_back(child);
      // add new_parent in same place as old_parent
      child->in_nodes.insert(child_in_nodes_position, new_parent);
    } else {
      i++;
    }
  }
}
} // namespace graph
} // namespace beanmachine
