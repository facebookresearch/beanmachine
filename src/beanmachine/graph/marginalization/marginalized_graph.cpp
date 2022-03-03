/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/marginalization/marginalized_graph.h"
#include "beanmachine/graph/marginalization/subgraph.h"

namespace beanmachine {
namespace graph {

MarginalizedGraph::MarginalizedGraph(Graph& g) : Graph(g) {
  // iterate over discrete variables and rewrite graph
}

void MarginalizedGraph::marginalize(uint discrete_sample_node_id) {
  Node* discrete_sample = get_node(discrete_sample_node_id);
  Node* discrete_distribution = discrete_sample->in_nodes[0];

  SubGraph subgraph = SubGraph(*this);

  // compute nodes up to and including stochastic children of discrete_sample
  std::set<uint> supp_ids = compute_full_support();
  std::vector<uint> det_node_ids;
  std::vector<uint> sto_node_ids;
  std::tie(det_node_ids, sto_node_ids) =
      compute_children(discrete_sample->index, supp_ids);

  // create MarginalizedDistribution

  // add nodes to subgraph

  // parents for MarginalizedDistribution

  // children for MarginalizedDistribution

  // create "COPY" for parents and children

  // move nodes to subgraph and finalize
}
} // namespace graph
} // namespace beanmachine
