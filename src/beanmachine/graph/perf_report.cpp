// Copyright (c) Facebook, Inc. and its affiliates.
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stack>
#include <string>

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
  void member(std::string name, std::string value) {
    comma();
    os << "\"" << name << "\" : \"" << value << "\"";
  }

  void member(std::string name, system_clock::time_point value) {
    auto t = system_clock::to_time_t(value);
    auto lt = localtime(&t);
    auto timestamp = std::put_time(lt, "%Y-%m-%d %H:%M:%S");
    comma();
    os << "\"" << name << "\" : \"" << timestamp << "\"";
  }

  void member(std::string name, uint value) {
    comma();
    os << "\"" << name << "\" : " << value << "\n";
  }

 private:
  void comma() {
    if (needs_comma.top()) {
      os << ",\n";
    } else {
      needs_comma.pop();
      needs_comma.push(true);
    }
  }
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
  js.member("title", "Bean Machine Graph performance report");
  js.member("generated_at", system_clock::now());
  js.member("num_samples", num_samples);
  js.member("algorithm", (uint)algorithm);
  js.member("seed", seed);
  js.member("node_count", nodes.size());
  js.member("edge_count", edge_count);
  js.end_object();
  _performance_report = js.str();
}
} // namespace graph
} // namespace beanmachine
