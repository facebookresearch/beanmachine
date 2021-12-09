/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <map>
#include <stack>
#include <vector>

namespace beanmachine {
namespace graph {

enum class ProfilerEvent {
  NMC_INFER,
  NMC_INFER_INITIALIZE,
  NMC_INFER_COLLECT_SAMPLES,
  NMC_INFER_COLLECT_SAMPLE,
  NMC_STEP,
  NMC_STEP_DIRICHLET,
  NMC_COMPUTE_GRADS,
  NMC_EVAL,
  NMC_CLEAR_GRADS,
  NMC_CREATE_PROP,
  NMC_CREATE_PROP_DIR,
  NMC_SAMPLE,
  NMC_SAVE_OLD,
  NMC_RESTORE_OLD,
};

struct Event {
  bool begin;
  ProfilerEvent kind;
  std::chrono::high_resolution_clock::time_point timestamp;
};

struct ProfilerData {
  std::vector<Event> events;
  std::stack<ProfilerEvent> in_flight;
  // Map from node id of a stochastic node to the number of
  // deterministic descendent nodes in the support.
  std::map<unsigned int, unsigned int> det_supp_count;
  ProfilerData();
  void begin(ProfilerEvent kind);
  void finish(ProfilerEvent kind);
};

} // namespace graph
} // namespace beanmachine
