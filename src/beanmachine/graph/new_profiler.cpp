/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format on

#include "beanmachine/graph/new_profiler.h"
#include <stdexcept>

#define NOW std::chrono::high_resolution_clock::now()

namespace beanmachine {
namespace graph {

Profiler* profiler = new Profiler();

Profiler::Profiler() {
  active = false;
  events.push_front(Event(first_event_start, NOW));
  id_generator = 2;
  event_names.push_back(std::string(""));
  event_names.push_back(std::string(""));
}

Profiler::~Profiler() {}

void Profiler::activate() {
  active = true;
}

void Profiler::deactivate() {
  active = false;
}

void Profiler::start(std::string event_name) { // inlined for speed
  if (active) {
    EventId_t start_event;
    EventId_t stop_event;

    auto monitored_event = monitored_events.find(event_name);

    if (monitored_event == monitored_events.end()) {
      start_event = id_generator++;
      stop_event = id_generator++;
      monitored_events[event_name] = EventIds_t(start_event, stop_event);
      event_names.push_back(event_name); // start
      event_names.push_back(event_name); // end
    } else {
      start_event = monitored_event->second.first;
    }
    Event event(start_event, NOW);
    events.push_front(event);
  }
}

void Profiler::stop(std::string event_name) {
  if (active) {
    int stop_event;

    auto monitored_event = monitored_events.find(event_name);

    if (monitored_event == monitored_events.end()) {
      throw(std::runtime_error(badstop_exception + event_name));
    } else {
      stop_event = monitored_event->second.second;
    }
    Event event(stop_event, NOW);
    events.push_front(event);
  }
}

void Profiler::get_report(std::string file_name) {
  std::ofstream stat_file(file_name, std::ios_base::out);
  if (!stat_file.is_open()) {
    throw(std::runtime_error(badfile_exception));
  }

  char now_str[27];
  now_str[26] = '\0';
  time_t now = time(nullptr);
  stat_file << "Title: Bean Machine Graph Profiler Report" << std::endl;
#ifdef _MSC_VER
  ctime_s(now_str, 26, &now);
#else
  ctime_r(&now, now_str);
#endif

  stat_file << "Generated at: " << now_str;

  // The first empty event brackets the list,
  // making a multi-rooted call tree into a single-rooted one

  events.push_front(Event(first_event_stop, NOW));
  events.reverse(); // can be eliminated if we switch to a vector
  check_integrity(stat_file);

  CallTreeNode_t* call_tree = construct_tree(nullptr);

  std::string tabs("");
  print_report(stat_file, call_tree, tabs);
}

// construct_tree is called after checking the integrity of the list
// therefore, we don't to do any integrity checks here, simplifying the code

Profiler::CallTreeNode_t* Profiler::construct_tree(CallTreeNode_t* node) {
  Event_t event = events.front(); // A start of an event
  events.pop_front();

  if (node == nullptr) {
    node = new (CallTreeNode_t);
    node->my_id = event.event_id;
    node->my_time = 0;
    node->num_calls = 0;
  }
  node->num_calls++;

  while (!events.empty()) {
    Event_t next_event = events.front();
    EventId_t next_evntid = next_event.event_id;
    if (next_evntid == node->my_id + 1) {
      events.pop_front();
      node->my_time += to_ticks(next_event.time_stamp, event.time_stamp);
      break;
    } else if (node->children.find(next_evntid) == node->children.end()) {
      node->children[next_evntid] = construct_tree(nullptr);
    } else {
      node->children[next_evntid] = construct_tree(node->children[next_evntid]);
    }
  }
  return node;
}

void Profiler::print_report(
    std::ofstream& stat_file,
    CallTreeNode_t* node,
    std::string tabs) {
  if (node->my_id == first_event_start) {
    stat_file << "List of profiled events:" << std::endl;
  } else if (node->my_id == first_event_stop) {
    // do nothing
  } else {
    stat_file << tabs << event_names[node->my_id] << ":[";
    stat_file << node->num_calls << "] ";
    stat_file << std::fixed << to_ms(node->my_time) << "ms" << std::endl;
  }
  for (auto it : node->children) {
    print_report(stat_file, it.second, std::string("   ") + tabs);
  }
  delete node;
}

double Profiler::to_ms(long long ticks) {
  return double(ticks) / double(TICKS_PER_MSEC);
}

long long Profiler::to_ticks(Stamp_t end, Stamp_t start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
      .count();
}

void Profiler::check_integrity(std::ofstream& stat_file) {
  // We have two types of errors:
  // a. Starts and stops of an event are mismatched (they must be equal)
  // b. Starts and stops are matched, but not nested properly

  std::map<std::string, std::pair<int, int>> start_end;
  for (auto it : events) {
    std::map<std::string, std::pair<int, int>>::const_iterator ev;
    ev = start_end.find(event_names[it.event_id]);
    if (ev == start_end.end()) {
      start_end[event_names[it.event_id]] = std::pair<int, int>(0, 0);
    }
    if (it.event_id & 0x1) {
      start_end[event_names[it.event_id]].second++; // stops
    } else {
      start_end[event_names[it.event_id]].first++; // starts
    }
  }
  stat_file << "Checking integrity of profiled event list" << std::endl;
  bool error = false;
  for (auto it : start_end) {
    if (it.second.first != it.second.second) {
      stat_file << "Event " << it.first << " has a mismatch: ";
      stat_file << it.second.first << " starts and ";
      stat_file << it.second.second << " stops." << std::endl;
      error = true;
    }
  }

  if (error) {
    stat_file << "Mismatched events in list. Bailing out" << std::endl;
    stat_file.close();
    throw(std::runtime_error(failed_integrity));
  } else {
    stat_file << "No mismatched events" << std::endl;
  }

  // Now check the nesting rules
  std::stack<Event_t> checker;
  for (auto it : events) {
    if ((it.event_id & 0x1) == 0) {
      checker.push(it);
    } else if (checker.top().event_id == (it.event_id - 1)) {
      checker.pop();
    } else {
      stat_file << "User error: Event: " << event_names[it.event_id];
      stat_file << " stopped while other events going. Nesting rule violation."
                << std::endl;
      stat_file << "Found error. Bailing out" << std::endl;
      stat_file.close();
      throw(std::runtime_error(notnested_exception));
    }
  }
  stat_file << "Integrity check complete. No errors" << std::endl;
}

} // namespace graph
} // namespace beanmachine
