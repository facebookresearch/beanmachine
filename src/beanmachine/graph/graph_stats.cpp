/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <beanmachine/graph/third-party/nameof.h>
#include <iomanip>
#include <stdexcept>
#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/factor/factor.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"

namespace beanmachine {
namespace graph {

// Public interface
std::string Graph::collect_statistics() {
  Statistics stats(*this);
  return stats.to_string();
}

std::string Graph::Statistics::to_string() {
  gen_graph_stats_report();
  gen_node_stats_report();
  gen_graph_properties_report();
  return report.str();
}

// end of public interface

// Private implementation

Graph::Statistics::Statistics(Graph& g) {
  initialize_scalars();
  initialize_count_vectors();
  compute_statistics(g);
}

// State initialization
void Graph::Statistics::initialize_scalars() {
  num_edges = 0;
  num_nodes = 0;
  max_in = 0;
  max_out = 0;
  graph_density = "";
  num_root_nodes = 0;
  num_terminal_nodes = 0;
}

void Graph::Statistics::initialize_count_vectors() {
  NodeType node_type(NodeType::UNKNOWN);
  OperatorType op_type(OperatorType::UNKNOWN);
  DistributionType dist_type(DistributionType::UNKNOWN);
  FactorType fact_type(FactorType::UNKNOWN);
  AtomicType atom_type(AtomicType::UNKNOWN);
  VariableType var_type(VariableType::UNKNOWN);

  node_type_counts.resize(ENUM_SIZE(node_type), 0);
  oper_counts.resize(ENUM_SIZE(op_type), 0);
  dist_counts.resize(ENUM_SIZE(dist_type), 0);
  fact_counts.resize(ENUM_SIZE(fact_type), 0);
  const_counts.resize(ENUM_SIZE(atom_type));

  for (uint i = 0; i < const_counts.size(); i++) {
    std::vector<uint> values;
    values.resize(ENUM_SIZE(var_type), 0);
    const_counts[i] = values;
  }

  in_edge_histogram.resize(1, 0);
  out_edge_histogram.resize(1, 0);
}

// statistics gathering methods

void Graph::Statistics::compute_statistics(Graph& g) {
  num_nodes = uint(g.nodes.size());
  for (auto const& node : g.nodes) {
    uint parents = uint((node.get())->in_nodes.size());
    uint children = uint((node.get())->out_nodes.size());
    parents == 0 ? num_root_nodes++ : 0;
    if (parents > max_in) {
      in_edge_histogram.resize((max_in = parents) + 1, 0);
    }
    parents <= in_edge_histogram.size() ? in_edge_histogram[parents]++ : 0;
    children == 0 ? num_terminal_nodes++ : 0;
    if (children > max_out) {
      out_edge_histogram.resize((max_out = children) + 1, 0);
    }
    children <= out_edge_histogram.size() ? out_edge_histogram[children]++ : 0;
    num_edges += parents;
    uint n_type = uint(node->node_type);
    n_type < node_type_counts.size() ? node_type_counts[n_type]++ : 0;
    compute_node_statistics(node->node_type, node.get());
  }
  graph_density = compute_density();
}

std::string Graph::Statistics::compute_density() {
  double max_density = double(num_nodes) * (double(num_nodes) - 1.0);
  double graph_density = double(num_edges) / max_density;
  Stream_t s_density;
  s_density << std::fixed;
  s_density << std::setprecision(2);
  s_density << graph_density;
  return s_density.str();
}

void Graph::Statistics::compute_node_statistics(NodeType n_type, Node* node) {
  if (n_type == NodeType::OPERATOR) {
    auto op = static_cast<oper::Operator*>(node);
    assert(uint(op->op_type) < oper_counts.size());
    oper_counts[uint(op->op_type)]++;
  } else if (n_type == NodeType::DISTRIBUTION) {
    auto dist = static_cast<distribution::Distribution*>(node);
    assert(uint(dist->dist_type) < dist_counts.size());
    dist_counts[uint(dist->dist_type)]++;
  } else if (n_type == NodeType::CONSTANT) {
    assert(uint(node->value.type.atomic_type) < const_counts.size());
    assert(
        uint(node->value.type.variable_type) <
        const_counts[uint(node->value.type.atomic_type)].size());
    const_counts[uint(node->value.type.atomic_type)]
                [uint(node->value.type.variable_type)]++;
  } else if (n_type == NodeType::FACTOR) {
    auto factor = static_cast<factor::Factor*>(node);
    assert(uint(factor->fac_type) < fact_counts.size());
    fact_counts[uint(factor->fac_type)]++;
  }
}

// Reoort generation methods

void Graph::Statistics::gen_graph_stats_report() {
  emit("Graph Statistics Report", '#');
  emit("Number of nodes", num_nodes);
  emit("Number of edges", num_edges);
  emit("Graph density", graph_density);
  emit();
}

void Graph::Statistics::gen_node_stats_report() {
  emit("Node statistics:", '#');
  for (uint i = 0; i < node_type_counts.size(); i++) {
    if (node_type_counts[i] > 0) {
      emit(String_t(NAMEOF_ENUM(NodeType(i))), node_type_counts[i]);
    }
  }
  emit();
  if (node_type_counts[uint(NodeType::OPERATOR)] > 0) {
    gen_detailed_stats<enum OperatorType>("Operator", oper_counts);
  }
  if (node_type_counts[uint(NodeType::DISTRIBUTION)] > 0) {
    gen_detailed_stats<enum DistributionType>("Distribution", dist_counts);
  }
  if (node_type_counts[uint(NodeType::FACTOR)] > 0) {
    gen_detailed_stats<enum FactorType>("Factor", fact_counts);
  }
  if (node_type_counts[uint(NodeType::CONSTANT)] > 0) {
    emit("Constant node statistics:", '#');
    for (uint i = 0; i < const_counts.size(); i++) {
      for (uint j = 0; j < const_counts[i].size(); j++) {
        if (const_counts[i][j] > 0) {
          String_t label;
          label = String_t(NAMEOF_ENUM(AtomicType(i)));
          label += " and ";
          label += String_t(NAMEOF_ENUM(VariableType(j)));
          emit(label, const_counts[i][j]);
        }
      }
    }
    emit();
  }
}

template <class T>
void Graph::Statistics::gen_detailed_stats(String_t title, Counts_t counts) {
  emit(title + " node statistics:", '#');
  for (uint i = 0; i < counts.size(); i++) {
    if (counts[i] > 0) {
      emit(String_t(NAMEOF_ENUM(T(i))), counts[i]);
    }
  }
  emit();
}

void Graph::Statistics::gen_graph_properties_report() {
  emit("Some graph properties:", '#');
  emit("Number of root nodes", num_root_nodes);
  emit("Number of terminal nodes", num_terminal_nodes);
  emit("Maximum number of incoming edges into a node", max_in);
  emit("Maximum number of outgoing edges from a node", max_out);
  gen_edge_stats_report("incoming", in_edge_histogram);
  gen_edge_stats_report("outgoing", out_edge_histogram);
}

void Graph::Statistics::gen_edge_stats_report(String_t etype, Counts_t counts) {
  emit("Distribution of " + etype + " edges:");
  for (uint i = 0; i < counts.size(); i++) {
    if (counts[i] > 0) {
      emit_tab();
      emit("Nodes with " + std::to_string(i) + " edges", counts[i]);
    }
  }
}

void Graph::Statistics::emit(String_t label, String_t value) {
  report << label << ": " << value << std::endl;
}

void Graph::Statistics::emit(String_t label, uint value) {
  report << label << ": " << std::to_string(value) << std::endl;
}

void Graph::Statistics::emit(String_t title) {
  report << title << std::endl;
}

void Graph::Statistics::emit() {
  report << std::endl;
}

void Graph::Statistics::emit(String_t title, char banner) {
  String_t line;
  line.resize(title.size(), banner);
  report << title << std::endl << line << std::endl;
}

void Graph::Statistics::emit_tab() {
  report << "\t";
}

} // namespace graph
} // namespace beanmachine
