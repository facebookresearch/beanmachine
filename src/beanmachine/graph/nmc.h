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
#include "beanmachine/graph/stepper/nmc_dirichlet_beta_single_site_stepper.h"
#include "beanmachine/graph/stepper/nmc_dirichlet_gamma_single_site_stepper.h"
#include "beanmachine/graph/stepper/nmc_scalar_single_site_stepper.h"
#include "beanmachine/graph/util.h"

#define NATURAL_TYPE unsigned long long int

namespace beanmachine {
namespace graph {

class NMC {
 public:
  virtual ~NMC();

 private:
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
  std::vector<NodeValue> old_values;

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

  // Single-site steppers application to nodes in unobserved_supp that are
  // stochastic; similarly, order matters.
  std::vector<NMCSingleSiteStepper*> stepper_for_node;

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
  // TODO: review what really needs to be private or public in NMC's API.
  // To do this, it may help to think of this class as an "enriched Graph",
  // since it contains graph but adds NMC-useful operations to it.

  std::mt19937 gen;

  NMC(Graph* g, uint seed);

  void infer(uint num_samples, InferConfig infer_config);

  void initialize();

  void collect_node_ptrs();

  void compute_support();

  void find_steppers();

  static bool is_not_supported(Node* node);

  void ensure_continuous();

  void compute_initial_values();

  void compute_affected_nodes();

  void generate_sample();

  NMCSingleSiteStepper* find_applicable_stepper(Node* tgt_node);

  void collect_samples(uint num_samples, InferConfig infer_config);

  void collect_sample(InferConfig infer_config);

  void save_old_values(const std::vector<Node*>& det_nodes);

  void restore_old_values(const std::vector<Node*>& det_nodes);

  void compute_gradients(const std::vector<Node*>& det_nodes);

  void eval(const std::vector<Node*>& det_nodes);

  void clear_gradients(const std::vector<Node*>& det_nodes);

  double compute_log_prob_of(const std::vector<Node*>& sto_nodes);

  NodeValue sample(const std::unique_ptr<proposer::Proposer>& prop);

  // TEMPORARY: we are in the process of moving ad hoc code out of NMC.
  // Eventually NMC will receive an arbitray list of single-site steppers that
  // will known to which random variables they apply.
  std::vector<NMCSingleSiteStepper*> single_site_steppers;
};

} // namespace graph
} // namespace beanmachine
