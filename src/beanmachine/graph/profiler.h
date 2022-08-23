/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <chrono>
#include <climits>
#include <cstring>
#include <ctime>
#include <exception>
#include <forward_list>
#include <fstream>
#include <iomanip>
#include <map>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

namespace beanmachine {
namespace graph {

class Profiler {
 public:
  Profiler();
  ~Profiler();
  void start(std::string event_name);
  void stop(std::string event_name);
  void activate();
  void deactivate();
  void get_report(std::string file_name);
  // end of the profiler interface

  std::string badstop_exception = "a stop event with no matching Start: ";
  std::string badfile_exception = "failed to open Stat file for writing";
  std::string notnested_exception = "profiler calls not nested properly?";
  std::string failed_integrity = "mismatch in events, see file";
  // end of the profiler exceptions

  /* ---------------------------------------------------------------*/
 private:
  // private types
  using EventId_t = unsigned int;
  using EventIds_t = std::pair<EventId_t, EventId_t>; // start & stop
  using Stamp_t = std::chrono::high_resolution_clock::time_point;
  using Event_t = struct Event {
    EventId_t event_id; // Start or stop index
    Stamp_t time_stamp;
    Event(EventId_t id, Stamp_t ts) : event_id(id), time_stamp(ts) {}
  };
  using CallTreeNode_t = struct CallTreeNode {
    std::map<EventId_t, struct CallTreeNode*> children;
    EventId_t my_id;
    long long my_time;
    unsigned num_calls;
  };
  // private data
  bool active;
  const double TICKS_PER_MSEC = 1000000.0;
  EventId_t id_generator;
  const EventId_t first_event_start = 0;
  const EventId_t first_event_stop = 1;
  std::unordered_map<std::string, EventIds_t> monitored_events;
  std::vector<std::string> event_names;
  std::forward_list<Event> events; // A vector is also plausible here
  // private methods
  void check_integrity(std::ofstream& stat_file);
  CallTreeNode_t* construct_tree(CallTreeNode* my_node);
  void print_report(std::ofstream& file, CallTreeNode* node, std::string tabs);
  // various helpers
  double to_ms(long long tks);
  long long to_ticks(Stamp_t end, Stamp_t start);
};

extern Profiler* profiler;

/*
The folliwing is the old profiler code. I am leaving it here for the moment
because some code in the compiler side depends on it, and I do not want to break
anything In the future, the new profiler should be made backward compatible with
the old one so that we do not need to have both. #pragma once
*/

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
