/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stack>
#include <string>
#include <variant>

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"

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
    struct tm lt;
#ifdef _MSC_VER
    localtime_s(&lt, &t);
#else
    localtime_r(&t, &lt);
#endif
    auto timestamp = std::put_time(&lt, "%Y-%m-%d %H:%M:%S");
    os << "\"" << timestamp << "\"";
  }

  void ticks(high_resolution_clock::time_point v) {
    number(static_cast<long>(v.time_since_epoch().count()));
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
      case ProfilerEvent::NMC_STEP:
        text("step");
        break;
      case ProfilerEvent::NMC_STEP_DIRICHLET:
        text("step_dirichlet");
        break;
      case ProfilerEvent::NMC_COMPUTE_GRADS:
        text("compute_grads");
        break;
      case ProfilerEvent::NMC_EVAL:
        text("eval");
        break;
      case ProfilerEvent::NMC_CLEAR_GRADS:
        text("clear_grads");
        break;
      case ProfilerEvent::NMC_CREATE_PROP:
        text("create_prop");
        break;
      case ProfilerEvent::NMC_CREATE_PROP_DIR:
        text("create_prop_dir");
        break;
      case ProfilerEvent::NMC_SAMPLE:
        text("sample");
        break;
      case ProfilerEvent::NMC_INFER_COLLECT_SAMPLE:
        text("collect_sample");
        break;
      case ProfilerEvent::NMC_SAVE_OLD:
        text("save_old");
        break;
      case ProfilerEvent::NMC_RESTORE_OLD:
        text("restore_old");
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
  uint factor_count = 0;
  uint const_count = 0;
  uint dist_count = 0;
  uint op_count = 0;
  uint add_count = 0;

  for (uint node_id = 0; node_id < static_cast<uint>(nodes.size()); node_id++) {
    Node* node = nodes[node_id].get();
    edge_count += static_cast<uint>(node->in_nodes.size());
    switch (node->node_type) {
      case NodeType::FACTOR:
        factor_count += 1;
        break;
      case NodeType::CONSTANT:
        const_count += 1;
        break;
      case NodeType::DISTRIBUTION:
        dist_count += 1;
        break;
      case NodeType::OPERATOR: {
        op_count += 1;
        auto op = static_cast<oper::Operator*>(node);
        if (op->op_type == OperatorType::ADD) {
          add_count += 1;
        }
        break;
      }
      default:
        break;
    }
  }

  js.start_object();
  js.text("title", "Bean Machine Graph performance report");
  js.date("generated_at", system_clock::now());
  js.number("num_samples", static_cast<long>(num_samples));
  js.number("algorithm", static_cast<uint>(algorithm));
  js.number("seed", seed);
  js.number("node_count", static_cast<long>(nodes.size()));
  js.number("edge_count", static_cast<long>(edge_count));
  js.number("factor_count", static_cast<long>(factor_count));
  js.number("dist_count", static_cast<long>(dist_count));
  js.number("const_count", static_cast<long>(const_count));
  js.number("op_count", static_cast<long>(op_count));
  js.number("add_count", static_cast<long>(add_count));

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
  js.member("det_supp_count");
  js.start_array();
  for (auto pair : profiler_data.det_supp_count) {
    js.comma();
    js.number(pair.second);
  }
  js.end_array();
  js.end_object();
  _performance_report = js.str();
}
} // namespace graph
} // namespace beanmachine
