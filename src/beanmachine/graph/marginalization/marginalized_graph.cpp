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
  std::set<uint> supp_ids = compute_full_support();
  std::vector<uint> det_node_ids;
  std::vector<uint> sto_node_ids;
  std::tie(det_node_ids, sto_node_ids) =
      compute_children(discrete_sample->index, supp_ids);

  // create MarginalizedDistribution
  std::unique_ptr<distribution::DummyMarginal> marginalized_node_pointer =
      std::make_unique<distribution::DummyMarginal>(std::move(subgraph_ptr));
  marginalized_node_pointer.get()->sample_type = AtomicType::REAL;
  Node* marginalized_node = marginalized_node_pointer.get();
  nodes.push_back(std::move(marginalized_node_pointer));

  // add nodes to subgraph
  // add discrete distribution and samples
  subgraph->add_node_by_id(discrete_distribution->index);
  subgraph->add_node_by_id(discrete_sample->index);
  // add all intermediate deterministic nodes to subgraph
  for (uint id : det_node_ids) {
    subgraph->add_node_by_id(id);
  }

  // parents for MarginalizedDistribution

  // children for MarginalizedDistribution

  // create "COPY" for parents and children

  // move nodes to subgraph and finalize
  subgraph->move_nodes_from_graph();
}
} // namespace graph
} // namespace beanmachine
