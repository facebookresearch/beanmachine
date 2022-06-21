/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
  // The stepper responsible for taking steps over the Markov chain.
  // Owned by this class; destructor deletes it.
  Stepper* stepper;

 public:
  Graph* graph;

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
  MH(Graph* graph, unsigned int seed, Stepper* stepper);

  void infer(uint num_samples, InferConfig infer_config);

  void initialize();

  void ensure_all_nodes_are_supported();

  void compute_initial_values();

  void generate_sample();

  SingleSiteSteppingMethod* find_applicable_single_site_stepping_method(
      Node* tgt_node);

  void collect_samples(uint num_samples, InferConfig infer_config);

  void collect_sample(InferConfig infer_config);

  NodeValue sample(const std::unique_ptr<proposer::Proposer>& prop);

  virtual ~MH();
};

} // namespace graph
} // namespace beanmachine
