/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/marginalization/marginalized_graph.h"
#include "beanmachine/graph/distribution/dummy_marginal.h"
#include "beanmachine/graph/graph.h"
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
    marginal_distribution->out_nodes.push_back(get_node(id));
  }

  // create "COPY" for parents and children

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
} // namespace graph
} // namespace beanmachine
