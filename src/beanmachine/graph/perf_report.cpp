// Copyright (c) Facebook, Inc. and its affiliates.
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stack>
#include <string>
#include <variant>

#include "beanmachine/graph/graph.h"

using namespace std::chrono;

namespace beanmachine {
namespace graph {

class JSON {
 public:
  std::string str() {
    return os.str();
  }

  void start_object() {
    os << "{\n";
    needs_comma.push(false);
  }

  void end_object() {
    os << "\n}";
    needs_comma.pop();
  }

  void start_array() {
    os << "[\n";
    needs_comma.push(false);
  }

  void end_array() {
    os << "\n]";
    needs_comma.pop();
  }

  void text(std::string v) {
    os << "\"" << v << "\"";
  }

  void boolean(bool v) {
    os << (v ? "true" : "false");
  }

  void date(system_clock::time_point v) {
    auto t = system_clock::to_time_t(v);
    auto lt = localtime(&t);
    auto timestamp = std::put_time(lt, "%Y-%m-%d %H:%M:%S");
    os << "\"" << timestamp << "\"";
  }

  void ticks(high_resolution_clock::time_point v) {
    number(v.time_since_epoch().count());
  }

  void number(long v) {
    os << v;
  }

  void member(std::string name) {
    comma();
    os << "\"" << name << "\" : ";
  }

  void boolean(std::string name, bool v) {
    member(name);
    boolean(v);
  }

  void text(std::string name, std::string v) {
    member(name);
    text(v);
  }

  void date(std::string name, system_clock::time_point v) {
    member(name);
    date(v);
  }

  void ticks(std::string name, high_resolution_clock::time_point v) {
    member(name);
    ticks(v);
  }

  void number(std::string name, long v) {
    member(name);
    number(v);
  }

  void event(ProfilerEvent v) {
    switch (v) {
      case ProfilerEvent::NMC_INFER:
        text("nmc_infer");
        break;
      case ProfilerEvent::NMC_INFER_COLLECT_SAMPLES:
        text("collect_samples");
        break;
      case ProfilerEvent::NMC_INFER_INITIALIZE:
        text("initialize");
        break;
      default:
        number((int)v);
        break;
    }
  }

  void event(std::string name, ProfilerEvent v) {
    member(name);
    event(v);
  }

  void comma() {
    if (needs_comma.top()) {
      os << ",\n";
    } else {
      needs_comma.pop();
      needs_comma.push(true);
    }
  }

 private:
  std::ostringstream os;
  std::stack<bool> needs_comma;
};

void Graph::collect_performance_data(bool b) {
  _collect_performance_data = b;
}

std::string Graph::performance_report() {
  return _performance_report;
}

void Graph::_produce_performance_report(
    uint num_samples,
    InferenceType algorithm,
    uint seed) {
  _performance_report = "";
  JSON js;
  if (!_collect_performance_data)
    return;

  uint edge_count = 0;
  for (uint node_id = 0; node_id < nodes.size(); node_id++) {
    edge_count += nodes[node_id]->in_nodes.size();
  }
  js.start_object();
  js.text("title", "Bean Machine Graph performance report");
  js.date("generated_at", system_clock::now());
  js.number("num_samples", num_samples);
  js.number("algorithm", (uint)algorithm);
  js.number("seed", seed);
  js.number("node_count", nodes.size());
  js.number("edge_count", edge_count);
  js.number("edge_count", edge_count);
  js.member("profiler_data");
  js.start_array();
  for (auto e : profiler_data.events) {
    js.comma();
    js.start_object();
    js.boolean("begin", e.begin);
    js.event("kind", e.kind);
    js.ticks("timestamp", e.timestamp);
    js.end_object();
  }

  js.end_array();
  js.end_object();
  _performance_report = js.str();
}
} // namespace graph
} // namespace beanmachine
