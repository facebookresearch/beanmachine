/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/mh.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/profiler.h"
#include "beanmachine/graph/proposer/default_initializer.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace graph {

MH::MH(Graph* graph, uint seed, Stepper* stepper)
    : unobserved_sto_support_index_by_node_id(graph->nodes.size(), 0),
      stepper(stepper),
      graph(graph),
      gen(seed) {}

void MH::infer(uint num_samples, InferConfig infer_config) {
  graph->pd_begin(ProfilerEvent::NMC_INFER);
  initialize();
  collect_samples(num_samples, infer_config);
  graph->pd_finish(ProfilerEvent::NMC_INFER);
}

// The initialization phase precomputes the vectors we are going to
// need during inference, and verifies that the MH algorithm can
// compute gradients of every node we need to.
void MH::initialize() {
  graph->pd_begin(ProfilerEvent::NMC_INFER_INITIALIZE);
  collect_node_ptrs();
  compute_support();
  ensure_all_nodes_are_supported();
  compute_initial_values();
  compute_affected_nodes();
  old_values = std::vector<NodeValue>(graph->nodes.size());
  graph->pd_finish(ProfilerEvent::NMC_INFER_INITIALIZE);
}

void MH::collect_node_ptrs() {
  for (uint node_id = 0; node_id < static_cast<uint>(graph->nodes.size());
       node_id++) {
    node_ptrs.push_back(graph->nodes[node_id].get());
  }
}

void MH::compute_support() {
  supp_ids = graph->compute_support();
  for (uint node_id : supp_ids) {
    supp.push_back(node_ptrs[node_id]);
  }
  for (Node* node : supp) {
    bool node_is_not_observed =
        graph->observed.find(node->index) == graph->observed.end();
    if (node_is_not_observed) {
      unobserved_supp.push_back(node);
      if (node->is_stochastic()) {
        uint index_of_next_unobserved_sto_supp_node =
            static_cast<uint>(unobserved_sto_supp.size());
        unobserved_sto_supp.push_back(node);
        uint node_id = node->index;
        unobserved_sto_support_index_by_node_id[node_id] =
            index_of_next_unobserved_sto_supp_node;
      }
    }
  }
}

void MH::ensure_all_nodes_are_supported() {
  for (Node* node : unobserved_sto_supp) {
    std::string error_message = is_not_supported(node);
    if (error_message != "") {
      throw std::runtime_error(error_message);
    }
  }
}

// We can now compute the initial state of the graph. Observed nodes
// will have values given by the observation, so we can ignore those.
// Unobserved stochastic nodes are assigned a value by the uniform
// initializer. Deterministic nodes are computed from their inputs.
// Note that we only need a single pass because parent nodes always have
// indices less than those of their children, and unobserved_supp
// respects index order.
void MH::compute_initial_values() {
  for (Node* unobs_node : unobserved_supp) {
    if (unobs_node->is_stochastic()) {
      proposer::default_initializer(gen, unobs_node);
    } else { // non-stochastic operator node, so just evaluate
      unobs_node->eval(gen);
    }
  }
}

// For every unobserved stochastic node in the graph, we will need to
// repeatedly know the set of immediate stochastic descendants
// and intervening deterministic nodes.
// Because this can be expensive, we compute those sets once and cache them.
void MH::compute_affected_nodes() {
  for (Node* node : unobserved_sto_supp) {
    std::vector<uint> det_node_ids;
    std::vector<uint> sto_node_ids;
    std::vector<Node*> det_nodes;
    std::vector<Node*> sto_nodes;
    std::tie(det_node_ids, sto_node_ids) =
        graph->compute_affected_nodes(node->index, supp_ids);
    for (uint id : det_node_ids) {
      det_nodes.push_back(node_ptrs[id]);
    }
    for (uint id : sto_node_ids) {
      sto_nodes.push_back(node_ptrs[id]);
    }
    det_affected_nodes.push_back(det_nodes);
    sto_affected_nodes.push_back(sto_nodes);
    if (graph->_collect_performance_data) {
      graph->profiler_data.det_supp_count[static_cast<uint>(node->index)] =
          static_cast<int>(det_nodes.size());
    }
  }
}

void MH::generate_sample() {
  stepper->step();
}

void MH::collect_samples(uint num_samples, InferConfig infer_config) {
  graph->pd_begin(ProfilerEvent::NMC_INFER_COLLECT_SAMPLES);
  for (uint snum = 0; snum < num_samples + infer_config.num_warmup; snum++) {
    generate_sample();
    if (infer_config.keep_warmup or snum >= infer_config.num_warmup) {
      collect_sample(infer_config);
    }
  }
  graph->pd_finish(ProfilerEvent::NMC_INFER_COLLECT_SAMPLES);
}

void MH::collect_sample(InferConfig infer_config) {
  graph->pd_begin(ProfilerEvent::NMC_INFER_COLLECT_SAMPLE);
  if (infer_config.keep_log_prob) {
    graph->collect_log_prob(graph->_full_log_prob(supp));
  }
  graph->collect_sample();
  graph->pd_finish(ProfilerEvent::NMC_INFER_COLLECT_SAMPLE);
}

const std::vector<Node*>& MH::get_det_affected_nodes(Node* node) {
  return det_affected_nodes
      [unobserved_sto_support_index_by_node_id[node->index]];
}

const std::vector<Node*>& MH::get_sto_affected_nodes(Node* node) {
  return sto_affected_nodes
      [unobserved_sto_support_index_by_node_id[node->index]];
}

void MH::revertibly_set_and_propagate(Node* node, const NodeValue& value) {
  save_old_value(node);
  save_old_values(get_det_affected_nodes(node));
  old_sto_affected_nodes_log_prob =
      compute_log_prob_of(get_sto_affected_nodes(node));
  node->value = value;
  eval(get_det_affected_nodes(node));
}

void MH::revert_set_and_propagate(Node* node) {
  restore_old_value(node);
  restore_old_values(get_det_affected_nodes(node));
}

void MH::save_old_value(const Node* node) {
  old_values[node->index] = node->value;
}

void MH::save_old_values(const std::vector<Node*>& nodes) {
  graph->pd_begin(ProfilerEvent::NMC_SAVE_OLD);
  for (Node* node : nodes) {
    old_values[node->index] = node->value;
  }
  graph->pd_finish(ProfilerEvent::NMC_SAVE_OLD);
}

NodeValue& MH::get_old_value(const Node* node) {
  return old_values[node->index];
}

void MH::restore_old_value(Node* node) {
  node->value = old_values[node->index];
}

void MH::restore_old_values(const std::vector<Node*>& det_nodes) {
  graph->pd_begin(ProfilerEvent::NMC_RESTORE_OLD);
  for (Node* node : det_nodes) {
    node->value = old_values[node->index];
  }
  graph->pd_finish(ProfilerEvent::NMC_RESTORE_OLD);
}

void MH::compute_gradients(const std::vector<Node*>& det_nodes) {
  graph->pd_begin(ProfilerEvent::NMC_COMPUTE_GRADS);
  for (Node* node : det_nodes) {
    node->compute_gradients();
  }
  graph->pd_finish(ProfilerEvent::NMC_COMPUTE_GRADS);
}

void MH::eval(const std::vector<Node*>& det_nodes) {
  graph->pd_begin(ProfilerEvent::NMC_EVAL);
  for (Node* node : det_nodes) {
    node->eval(gen);
  }
  graph->pd_finish(ProfilerEvent::NMC_EVAL);
}

void MH::clear_gradients(Node* node) {
  // TODO: eventually we want to have different classes of Node
  // and have this be a virtual method
  switch (node->value.type.variable_type) {
    case VariableType::SCALAR:
      node->grad1 = 0;
      node->grad2 = 0;
      break;
    case VariableType::BROADCAST_MATRIX:
    case VariableType::COL_SIMPLEX_MATRIX: {
      auto rows = node->value._matrix.rows();
      auto cols = node->value._matrix.cols();
      node->Grad1 = Eigen::MatrixXd::Zero(rows, cols);
      node->Grad2 = Eigen::MatrixXd::Zero(rows, cols);
      break;
    }
    default:
      throw std::runtime_error(
          "clear_gradients invoked for nodes of an unsupported variable type " +
          std::to_string(int(node->value.type.variable_type)));
  }
}

void MH::clear_gradients(const std::vector<Node*>& nodes) {
  graph->pd_begin(ProfilerEvent::NMC_CLEAR_GRADS);
  for (Node* node : nodes) {
    clear_gradients(node);
  }
  graph->pd_finish(ProfilerEvent::NMC_CLEAR_GRADS);
}

void MH::clear_gradients_of_node_and_its_affected_nodes(Node* node) {
  clear_gradients(node);
  clear_gradients(get_det_affected_nodes(node));
  clear_gradients(get_sto_affected_nodes(node));
}

// Computes the log probability with respect to a given
// set of stochastic nodes.
double MH::compute_log_prob_of(const std::vector<Node*>& sto_nodes) {
  double log_prob = 0;
  for (Node* node : sto_nodes) {
    log_prob += node->log_prob();
  }
  return log_prob;
}

NodeValue MH::sample(const std::unique_ptr<proposer::Proposer>& prop) {
  graph->pd_begin(ProfilerEvent::NMC_SAMPLE);
  NodeValue v = prop->sample(gen);
  graph->pd_finish(ProfilerEvent::NMC_SAMPLE);
  return v;
}

MH::~MH() {
  delete stepper;
}

} // namespace graph
} // namespace beanmachine
