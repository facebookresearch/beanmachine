// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include "beanmachine/graph/profiler.h"

using namespace std::chrono;

namespace beanmachine {
namespace graph {

ProfilerData::ProfilerData() {}

void ProfilerData::begin(ProfilerEvent kind) {
  auto t = high_resolution_clock::now();
  events.push_back({true, kind, t});
  in_flight.push(kind);
}

void ProfilerData::finish(ProfilerEvent kind) {
  auto t = high_resolution_clock::now();
  while (!in_flight.empty()) {
    ProfilerEvent top = in_flight.top();
    in_flight.pop();
    events.push_back({false, top, t});
    if (top == kind) {
      break;
    }
  }
}

} // namespace graph
} // namespace beanmachine
