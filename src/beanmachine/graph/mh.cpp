/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <boost/iostreams/stream.hpp>
#include <boost/progress.hpp>
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

namespace beanmachine {
namespace graph {

MH::MH(Graph* graph, uint seed, Stepper* stepper)
    : stepper(stepper), graph(graph), gen(seed) {}

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
  graph->ensure_evaluation_and_inference_readiness();
  ensure_all_nodes_are_supported();
  compute_initial_values();
}

void MH::ensure_all_nodes_are_supported() {
  for (Node* node : graph->unobserved_sto_supp) {
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
  for (Node* unobs_node : graph->unobserved_supp) {
    if (unobs_node->is_stochastic()) {
      proposer::default_initializer(gen, unobs_node);
    } else { // non-stochastic operator node, so just evaluate
      unobs_node->eval(gen);
    }
  }
}

void MH::generate_sample() {
  stepper->step();
}

void MH::collect_samples(uint num_samples, InferConfig infer_config) {
  graph->pd_begin(ProfilerEvent::NMC_INFER_COLLECT_SAMPLES);
  boost::iostreams::stream<boost::iostreams::null_sink> nullOstream(
      (boost::iostreams::null_sink()));
  boost::progress_display show_progress(
      num_samples, graph->thread_index == 0 ? std::cout : nullOstream);
  for (uint snum = 0; snum < num_samples + infer_config.num_warmup; snum++) {
    generate_sample();
    if (infer_config.keep_warmup or snum >= infer_config.num_warmup) {
      collect_sample(infer_config);
      if (graph->thread_index == 0) {
        ++show_progress;
      }
    }
  }
  graph->pd_finish(ProfilerEvent::NMC_INFER_COLLECT_SAMPLES);
}

void MH::collect_sample(InferConfig infer_config) {
  graph->pd_begin(ProfilerEvent::NMC_INFER_COLLECT_SAMPLE);
  if (infer_config.keep_log_prob) {
    graph->collect_log_prob(graph->full_log_prob());
  }
  graph->collect_sample();
  graph->pd_finish(ProfilerEvent::NMC_INFER_COLLECT_SAMPLE);
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
