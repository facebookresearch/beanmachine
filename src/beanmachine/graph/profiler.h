// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#pragma once

#include <chrono>
#include <stack>
#include <vector>

namespace beanmachine {
namespace graph {

enum class ProfilerEvent {
  NMC_INFER,
  NMC_INFER_INITIALIZE,
  NMC_INFER_COLLECT_SAMPLES,
  NMC_STEP,
  NMC_STEP_DIRICHLET,
};

struct Event {
  bool begin;
  ProfilerEvent kind;
  std::chrono::high_resolution_clock::time_point timestamp;
};

struct ProfilerData {
  std::vector<Event> events;
  std::stack<ProfilerEvent> in_flight;
  ProfilerData();
  void begin(ProfilerEvent kind);
  void finish(ProfilerEvent kind);
};

} // namespace graph
} // namespace beanmachine
