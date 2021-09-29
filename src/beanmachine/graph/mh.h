// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/profiler.h"
#include "beanmachine/graph/proposer/default_initializer.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/stepper/single_site/sequential_single_site_stepper.h"
#include "beanmachine/graph/util.h"

#define NATURAL_TYPE unsigned long long int

namespace beanmachine {
namespace graph {

class MH {
 protected:
  Graph* g;

  // A graph maintains of a vector of nodes; the index into that vector is
  // the id of the node. We often need to translate from node ids into node
  // pointers in this algorithm; to do so quickly we obtain the address of
  // every node in the graph up front and then look it up when we need it.
  std::vector<Node*> node_ptrs;

  // Every node in the graph has a value; when we propose a new graph state,
  // we update the values. If we then reject the proposed new state, we need
  // to restore the values. This vector stores the original values of the
  // nodes that we change during the proposal step.
  // We do the same for the log probability of the stochastic nodes
  // affected by the last revertible set and propagate operation
  // see (revertibly_set_and_propagate method).
  std::vector<NodeValue> old_values;
  double old_sto_affected_nodes_log_prob;

  // The support is the set of all nodes in the graph that are queried or
  // observed, directly or indirectly. We need both the support as nodes
  // and as pointers in this algorithm.
  std::set<uint> supp_ids;
  std::vector<Node*> supp;

  // Nodes in supp that are not directly observed. Note that
  // the order of nodes in this vector matters! We must enumerate
  // them in order from lowest node identifier to highest.
  std::vector<Node*> unobserved_supp;

  // Nodes in unobserved_supp that are stochastic; similarly, order matters.
  std::vector<Node*> unobserved_sto_supp;

  // A vector containing the index of a node in unobserved_sto_supp for each
  // node_id. Since not all nodes are in unobserved_sto_support, some elements
  // of this vector should never be accessed.
  std::vector<uint> unobserved_sto_support_index_by_node_id;

  // The stepper responsible for taking steps over the Markov chain.
  Stepper* stepper;

  // These vectors are the same size as unobserved_sto_support.
  // The i-th elements are vectors of nodes which are
  // respectively the vector of
  // the immediate stochastic descendants of node with index i in the support,
  // and the vector of the intervening deterministic nodes
  // between the i-th node and its immediate stochastic descendants.
  // In other words, these are the cached results of
  // invoking graph::compute_affected_nodes
  // for each node.
  std::vector<std::vector<Node*>> sto_affected_nodes;
  std::vector<std::vector<Node*>> det_affected_nodes;

 public:
  // Method testing whether a node is supported by algorithm.
  // It must return a non-empty string with an error message in case
  // the node is not supported.
  // TODO: this should be delegated to steppers, since that's
  // where this can really be decided.
  virtual std::string is_not_supported(Node* node) = 0;

  // TODO: review what really needs to be private or public in MH's API.
  // To do this, it may help to think of this class as an "enriched Graph",
  // since it contains graph but adds MH-useful operations to it.

  std::mt19937 gen;

  // Constructs MH algorithm based on stepper.
  // Takes ownership of stepper instance.
  MH(Graph* g, uint seed, Stepper* stepper);

  const std::vector<Node*>& unobserved_stochastic_support() {
    return unobserved_sto_supp;
  }

  void infer(uint num_samples, InferConfig infer_config);

  void initialize();

  void collect_node_ptrs();

  void compute_support();

  void ensure_all_nodes_are_supported();

  void compute_initial_values();

  void compute_affected_nodes();

  void generate_sample();

  SingleSiteSteppingMethod* find_applicable_single_site_stepping_method(
      Node* tgt_node);

  void collect_samples(uint num_samples, InferConfig infer_config);

  void collect_sample(InferConfig infer_config);

  const std::vector<Node*>& get_det_affected_nodes(Node* node);

  const std::vector<Node*>& get_sto_affected_nodes(Node* node);

  // Sets a given node to a new value and
  // updates its deterministically affected nodes.
  // Does so in a revertible manner by saving old values and old stochastic
  // affected nodes log prob.
  // Old values can be accessed through get_old_* methods.
  // The reversion is executed by invoking revert_set_and_propagate.
  void revertibly_set_and_propagate(Node* node, const NodeValue& value);

  // Revert the last revertibly_set_and_propagate
  void revert_set_and_propagate(Node* node);

  void save_old_value(const Node* node);

  void save_old_values(const std::vector<Node*>& nodes);

  NodeValue& get_old_value(const Node* node);

  double get_old_sto_affected_nodes_log_prob() {
    return old_sto_affected_nodes_log_prob;
  }

  void restore_old_value(Node* node);

  void restore_old_values(const std::vector<Node*>& det_nodes);

  void compute_gradients(const std::vector<Node*>& det_nodes);

  void eval(const std::vector<Node*>& det_nodes);

  void clear_gradients(Node* node);

  void clear_gradients(const std::vector<Node*>& nodes);

  void clear_gradients_of_node_and_its_affected_nodes(Node* node);

  double compute_log_prob_of(const std::vector<Node*>& sto_nodes);

  NodeValue sample(const std::unique_ptr<proposer::Proposer>& prop);

  virtual ~MH();
};

} // namespace graph
} // namespace beanmachine
