/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/marginalization/marginalized_graph.h"
#include <algorithm>
#include <memory>
#include "beanmachine/graph/distribution/dummy_marginal.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/marginalization/subgraph.h"
#include "beanmachine/graph/operator/stochasticop.h"

namespace beanmachine {
namespace graph {

std::tuple<std::vector<uint>, std::vector<uint>> compute_children(
    Graph& graph,
    uint node_id);

void add_nodes_to_subgraph(
    SubGraph* subgraph,
    Node* discrete_distribution,
    Node* discrete_sample,
    std::vector<uint> det_node_ids,
    std::vector<uint> sto_node_ids);

bool is_parent(Node* node, Node* parent);

void connect_parents_to_marginal_distribution(
    Graph& graph,
    distribution::DummyMarginal* marginal_distribution);

void move_children_of_first_node_to_second_node_if(
    Node* current_parent,
    Node* new_parent,
    std::function<bool(Node*)> condition);

void add_copy_of_parent_nodes_to_subgraph(
    SubGraph* subgraph,
    distribution::DummyMarginal* marginal_distribution);

std::vector<std::unique_ptr<Node>>
create_and_connect_children_to_marginal_distribution(
    Graph& graph,
    SubGraph* subgraph,
    distribution::DummyMarginal* marginal_distribution,
    std::vector<uint> sto_node_ids);

uint compute_largest_parent_index(Node* node);

/*
Creates a MarginalDistribution to replace the discrete node
and other affected nodes.

The MarginalDistribution has a stand-alone "subgraph", which will contain
all of the nodes required to compute the MarginalDistribution
1. the discrete distribution node
2. the discrete sample node
3. deterministic children nodes of the discrete sample up until
4. the stochastic children nodes of the discrete sample
5. the parents (a node not in 1-4 that has a child in 1-4)

The original graph will contain
1. the MarginalDistribution node (to replace #1-3 from the
   subgraph above)
2. the children of the MarginalDistribution are the
   stochastic children nodes of the discrete node
   (the same as #4 from the subgraph)
3. the parents of the MarginalDistribution are the parents
   of the subgraph (same as #5 from the subgraph)

In order to keep the original graph and the subgraph completely
independent, there are a few key points.

PARENTS:
The parents of the MarginalDistribution are the nodes not in the subgraph
that have a child in the subgraph.
To keep the subgraph as a "stand-alone" graph, we need to make sure that
it does not contain incoming nodes from the graph. To do this, we
we make a "copy" node in the subgraph for each of the parents in the graph.
This "copy" node in the subgraph is a Constant node whose value is the
same as the parent node in the graph.

CHILDREN:
The children of the MarginalDistribution are the stochastic children of
the discrete sample node.
The stochastic children are needed to compute the MarginalDistribution,
so they are part of the subgraph.
However, a "copy" of these children also needs to be added to the graph.
This "copy" node is a SAMPLE node of MarginalDistribution whose value
is the same as the stochastic child node in the subgraph.
*/
void marginalize_graph(Graph& graph, uint discrete_sample_node_id) {
  Node* discrete_sample = graph.get_node(discrete_sample_node_id);
  Node* discrete_distribution = discrete_sample->in_nodes[0];

  std::unique_ptr<SubGraph> subgraph_ptr = std::make_unique<SubGraph>(graph);

  // compute nodes up to and including stochastic children of discrete_sample
  std::vector<uint> det_node_ids;
  std::vector<uint> sto_node_ids;
  std::tie(det_node_ids, sto_node_ids) =
      compute_children(graph, discrete_sample->index);

  // create MarginalDistribution
  std::unique_ptr<distribution::DummyMarginal> marginal_distribution_ptr =
      std::make_unique<distribution::DummyMarginal>(std::move(subgraph_ptr));
  // TODO: support the correct sample type for multiple children
  if (sto_node_ids.size() > 0) {
    // @lint-ignore
    marginal_distribution_ptr->sample_type =
        graph.nodes[sto_node_ids[0]]->value.type; // NOLINT
  }
  // pointers for easier reference
  distribution::DummyMarginal* marginal_distribution =
      marginal_distribution_ptr.get();
  SubGraph* subgraph = marginal_distribution->subgraph_ptr.get();

  // add nodes to subgraph
  add_nodes_to_subgraph(
      subgraph,
      discrete_distribution,
      discrete_sample,
      det_node_ids,
      sto_node_ids);

  // connect parents to MarginalDistribution in graph
  connect_parents_to_marginal_distribution(graph, marginal_distribution);
  // add copy of parents to subgraph
  add_copy_of_parent_nodes_to_subgraph(subgraph, marginal_distribution);

  // create and connect children to MarginalDistribution
  std::vector<std::unique_ptr<Node>> created_children_nodes =
      create_and_connect_children_to_marginal_distribution(
          graph, subgraph, marginal_distribution, sto_node_ids);

  // list of all created nodes to add to `nodes` of current graph
  std::vector<std::unique_ptr<Node>> created_graph_nodes;
  // add MarginalDistribution to list of created nodes
  created_graph_nodes.push_back(std::move(marginal_distribution_ptr));
  // add created nodes to list of created_graph_nodes
  for (uint i = 0; i < created_children_nodes.size(); i++) {
    created_graph_nodes.push_back(std::move(created_children_nodes[i]));
  }

  // move the list of subgraph nodes from graph
  subgraph->move_nodes_from_graph_and_reindex();
  // add created nodes (MarginalDistribution and SAMPLE children) to the graph
  // NOTE: to keep the graph ordering invariant
  // (index of parents <= index of node),
  // the created nodes should be inserted right after the largest parent index
  uint marginal_distribution_index =
      compute_largest_parent_index(marginal_distribution) + 1;
  // insert created nodes into graph at "marginalized_node_index"
  for (uint i = 0; i < created_graph_nodes.size(); i++) {
    graph.nodes.insert(
        std::next(graph.nodes.begin(), marginal_distribution_index + i),
        std::move(created_graph_nodes[i]));
  }
  graph.reindex_nodes();
}

/*
returns <determinisitc_node_ids, stochastic_node_ids>
1. deterministic_node_ids are all of the deterministic nodes up until the
2. stochastic_node_ids children are reached
*/
std::tuple<std::vector<uint>, std::vector<uint>> compute_children(
    Graph& graph,
    uint node_id) {
  std::set<uint> supp_ids = graph.compute_full_ordered_support_node_ids();
  std::vector<uint> det_node_ids;
  std::vector<uint> sto_node_ids;
  return graph.compute_children(node_id, supp_ids);
}

/*
Add the necessary nodes to the "subgraph" to compute the
marginal distribution
This includes
1. the discrete distribution and the discrete sample
2. all of the intermediate deterministic nodes
3. all of the stochastic children of the discrete node
After this function, all of the nodes continue to exist
in the main "graph" and not the "subgraph". All graph
connections also exist in the main graph.
This method tells the "subgraph" to store the node ids
of the nodes that should be moved, and the nodes will
be moved at the end of the `marginalize_graph` method.
*/
void add_nodes_to_subgraph(
    SubGraph* subgraph,
    Node* discrete_distribution,
    Node* discrete_sample,
    std::vector<uint> det_node_ids,
    std::vector<uint> sto_node_ids) {
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
}

/*
is `parent` node in the parent list of `node`
*/
bool is_parent(Node* node, Node* parent) {
  return std::find(node->in_nodes.begin(), node->in_nodes.end(), parent) !=
      node->in_nodes.end();
}

/*
The parents of the marginal distribution should be a unique list
that includes all parents of any nodes inside the subgraph
which are not already part of the subgraph
*/
void connect_parents_to_marginal_distribution(
    Graph& graph,
    distribution::DummyMarginal* marginal_distribution) {
  auto& subgraph = marginal_distribution->subgraph_ptr;
  for (uint node_id : subgraph->get_node_ids()) {
    Node* subgraph_node = graph.get_node(node_id);
    for (Node* potential_parent : subgraph_node->in_nodes) {
      // check that potential_parent is not already a parent
      // check that potential_parent is not in the subgraph
      if (!is_parent(marginal_distribution, potential_parent) and
          !subgraph->has_node(potential_parent->index)) {
        marginal_distribution->in_nodes.push_back(potential_parent);
        potential_parent->out_nodes.push_back(marginal_distribution);
      }
    }
  }
}

/*
move the children from current_parent to new_parent
where condition(child) is true
*/
void move_children_of_first_node_to_second_node_if(
    Node* current_parent,
    Node* new_parent,
    std::function<bool(Node*)> condition) {
  // move children from current_parent to new_parent
  uint i = 0;
  while (i < current_parent->out_nodes.size()) {
    Node* child = current_parent->out_nodes[i];
    // only move children which meet condition
    if (condition(child)) {
      // replace child->current_parent with child->new_parent
      auto parent_position_in_child_in_nodes = std::find(
          child->in_nodes.begin(), child->in_nodes.end(), current_parent);
      *parent_position_in_child_in_nodes = new_parent;
      // remove current_parent->child
      current_parent->out_nodes.erase(current_parent->out_nodes.begin() + i);
      // add new_parent->child
      new_parent->out_nodes.push_back(child);
    } else {
      i++;
    }
  }
}

/*
The parents of the MarginalDistribution are already updated
(the nodes not in the subgraph that have children in the subgraph).
To keep the subgraph as a "stand-alone" graph, we add a "copy"
of each parent in the subgraph. This "copy" node is a constant node
whose value is the same as the graph parent value.

The children of these parents also need to be updated, where the
children of the nodes of in the graph should exist in the graph
and the children of the nodes in the subgraph should exist in the subgraph.
*/
void add_copy_of_parent_nodes_to_subgraph(
    SubGraph* subgraph,
    distribution::DummyMarginal* marginal_distribution) {
  for (Node* parent : marginal_distribution->in_nodes) {
    // create copy of parent nodes inside subgraph
    std::unique_ptr<ConstNode> parent_copy =
        std::make_unique<ConstNode>(parent->value);
    // TODO: add link between copy and parent in MarginalDistribution

    // the children of nodes in the graph should exist in the graph
    // and the children of nodes in the subgraph should exist in the subgraph

    // A child should be moved
    // from `parent` (in graph) to parent_copy (in subgraph)
    // if the child is in the subgraph
    move_children_of_first_node_to_second_node_if(
        parent, parent_copy.get(), [&](Node* child) {
          return subgraph->has_node(child->index);
        });
    subgraph->nodes.push_back(std::move(parent_copy));
  }
}

/*
Add the children of the marginal_distribution to the graph
where each child is a Sample node whose value is equivalent
to the value of the stochastic child in the subgraph

The stochastic children in the subgraph have two types of children:
1. children in the subgraph
2. children not in the subgraph

Let the current stochastic child node in the subgraph be C.
We want to create a copy of the child node in the graph, C'.

The children of C which are in the graph (not the subgraph)
should be moved to be children of C', since
the children of the nodes of in the graph should exist in the graph
and the children of the nodes in the subgraph should exist in the subgraph.

The children of C which are in the subgraph remain as children of C
and are not connected to C'.
*/
std::vector<std::unique_ptr<Node>>
create_and_connect_children_to_marginal_distribution(
    Graph& graph,
    SubGraph* subgraph,
    distribution::DummyMarginal* marginal_distribution,
    std::vector<uint> sto_node_ids) {
  std::vector<std::unique_ptr<Node>> created_nodes;

  // create copy of all of the discrete_sample's stochastic children
  for (uint id : sto_node_ids) {
    Node* child_node = graph.get_node(id);
    std::vector<graph::Node*> sample_in_nodes = {marginal_distribution};
    std::unique_ptr<oper::Sample> child_copy =
        std::make_unique<oper::Sample>(sample_in_nodes);
    // TODO: add link between copy and child in MarginalDistribution

    // child x should be moved
    // from original node (in subgraph) to sample_node (in graph)
    // if x is in the graph and therefore not in the subgraph
    move_children_of_first_node_to_second_node_if(
        child_node, child_copy.get(), [&](Node* child) {
          return !subgraph->has_node(child->index);
        });

    // connect sample_node as a child to marginal distribution
    marginal_distribution->out_nodes.push_back(child_copy.get());
    child_copy.get()->in_nodes.push_back(marginal_distribution);

    // add sample_node to list of created_nodes to add to graph
    created_nodes.push_back(std::move(child_copy));
  }
  return created_nodes;
}

/*
returns the largest index of a node's parents
*/
uint compute_largest_parent_index(Node* node) {
  uint largest_parent_index = 0;
  for (Node* parent : node->in_nodes) {
    // ordering invariant
    if (parent->index >= largest_parent_index) {
      largest_parent_index = parent->index;
    }
  }
  return largest_parent_index;
}

} // namespace graph
} // namespace beanmachine
