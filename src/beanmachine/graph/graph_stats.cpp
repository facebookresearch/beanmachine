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
  gen_edge_stats_report();
  return report.str();
}

// end of public interface

// Private implementation

Graph::Statistics::Statistics(Graph& g) {
  comp_graph_stats(g);
  comp_node_stats(g);
  comp_edge_stats(g);
}

// statistics gathering methods

void Graph::Statistics::comp_graph_stats(Graph& g) {
  num_edges = 0;
  max_in = 0;
  max_out = 0;
  num_root_nodes = 0;
  num_terminal_nodes = 0;
  num_nodes = uint(g.nodes.size());

  for (auto const& node : g.nodes) {
    uint parents = uint(node->in_nodes.size());
    parents > max_in ? max_in = parents : 0;
    parents == 0 ? num_root_nodes++ : 0;

    uint children = uint(node->out_nodes.size());
    children > max_out ? max_out = children : 0;
    children == 0 ? num_terminal_nodes++ : 0;

    num_edges += parents;
  }
  graph_density = compute_density();
}

void Graph::Statistics::comp_node_stats(Graph& g) {
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

  uint num_atomic = uint(ENUM_SIZE(atom_type));
  uint num_variable = uint(ENUM_SIZE(var_type));
  init_matrix(const_counts, num_atomic, num_variable);
  init_matrix(root_terminal_per_node_type, uint(ENUM_SIZE(node_type)), uint(2));

  for (auto const& node : g.nodes) {
    uint n_type = uint(node->node_type);
    node_type_counts[n_type]++;
    uint parents = uint(node->in_nodes.size());
    parents == 0 ? root_terminal_per_node_type[n_type][0]++ : 0;
    uint children = uint(node->out_nodes.size());
    children == 0 ? root_terminal_per_node_type[n_type][1]++ : 0;

    if (node->node_type == NodeType::OPERATOR) {
      auto op = static_cast<oper::Operator*>(node.get());
      oper_counts[uint(op->op_type)]++;
    } else if (node->node_type == NodeType::DISTRIBUTION) {
      auto dist = static_cast<distribution::Distribution*>(node.get());
      dist_counts[uint(dist->dist_type)]++;
    } else if (node->node_type == NodeType::FACTOR) {
      auto factor = static_cast<factor::Factor*>(node.get());
      fact_counts[uint(factor->fac_type)]++;
    } else if (node->node_type == NodeType::CONSTANT) {
      const_counts[uint(node->value.type.atomic_type)]
                  [uint(node->value.type.variable_type)]++;
    }
  }
}

void Graph::Statistics::comp_edge_stats(Graph& g) {
  NodeType node_type(NodeType::UNKNOWN);

  in_edge_histogram.resize(max_in + 1, 0);
  out_edge_histogram.resize(max_out + 1, 0);
  init_matrix(in_edge_bytype, uint(ENUM_SIZE(node_type)), max_in + 1);
  init_matrix(out_edge_bytype, uint(ENUM_SIZE(node_type)), max_out + 1);

  for (auto const& node : g.nodes) {
    uint n_type = uint(node->node_type);
    uint parents = uint(node->in_nodes.size());
    uint children = uint(node->out_nodes.size());

    in_edge_histogram[parents]++;
    out_edge_histogram[children]++;

    in_edge_bytype[n_type][parents]++;
    out_edge_bytype[n_type][children]++;
  }
}

void Graph::Statistics::init_matrix(Matrix_t& counts, uint rows, uint cols) {
  for (uint i = 0; i < rows; i++) {
    Counts_t values;
    values.resize(cols, 0);
    counts.resize(i + 1, values);
  }
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

// Reoort generation methods

void Graph::Statistics::gen_graph_stats_report() {
  tab = 0;
  emit("Graph Statistics Report", '#');
  emit("Number of nodes: " + std::to_string(num_nodes));
  emit("Number of edges: " + std::to_string(num_edges));
  emit("Graph density: " + graph_density);
  emit("Number of root nodes: " + std::to_string(num_root_nodes));
  emit("Number of terminal nodes: " + std::to_string(num_terminal_nodes));
  emit("Maximum no. of incoming edges into a node: " + std::to_string(max_in));
  emit("Maximum no. of outgoing edges from a node: " + std::to_string(max_out));
  emit("");
}

void Graph::Statistics::gen_node_stats_report() {
  tab = 0;
  emit("Node statistics:", '#');
  for (uint i = 0; i < uint(node_type_counts.size()); i++) {
    tab = 0;
    if (node_type_counts[i] > 0) {
      NodeType n_type = NodeType(i);
      String_t t_name = String_t(NAMEOF_ENUM(n_type));
      emit(t_name + ": " + std::to_string(node_type_counts[i]));
      gen_roots_and_terminals(i);

      switch (n_type) {
        case NodeType::OPERATOR:
          gen_operator_stats(oper_counts);
          break;
        case NodeType::DISTRIBUTION:
          gen_distribution_stats(dist_counts);
          break;
        case NodeType::FACTOR:
          gen_factor_stats(fact_counts);
          break;
        case NodeType::CONSTANT:
          gen_constant_stats(const_counts);
          break;
        default:
          emit("Unrecognized node type in statistics module");
          break;
      }
      gen_edge_stats_report("incoming", in_edge_bytype[i]);
      gen_edge_stats_report("outgoing", out_edge_bytype[i]);
    }
  }
}

void Graph::Statistics::gen_operator_stats(Counts_t counts) {
  emit("Operator node statistics:", '-');
  tab++;
  for (uint i = 0; i < uint(counts.size()); i++) {
    if (counts[i] > 0) {
      String_t label = String_t(NAMEOF_ENUM(OperatorType(i)));
      emit(label + ": " + std::to_string(counts[i]));
    }
  }
  emit("");
}

void Graph::Statistics::gen_distribution_stats(Counts_t counts) {
  emit("Distribution node statistics:", '-');
  tab++;
  for (uint i = 0; i < uint(counts.size()); i++) {
    if (counts[i] > 0) {
      String_t label = String_t(NAMEOF_ENUM(DistributionType(i)));
      emit(label + ": " + std::to_string(counts[i]));
    }
  }
  emit("");
}

void Graph::Statistics::gen_factor_stats(Counts_t counts) {
  emit("Factor node statistics:", '-');
  tab++;
  for (uint i = 0; i < uint(counts.size()); i++) {
    if (counts[i] > 0) {
      String_t label = String_t(NAMEOF_ENUM(FactorType(i)));
      emit(label + ": " + std::to_string(counts[i]));
    }
  }
  emit("");
}

void Graph::Statistics::gen_constant_stats(Matrix_t counts) {
  emit("Constant node statistics:", '-');
  tab++;
  for (uint k = 0; k < uint(counts.size()); k++) {
    for (uint l = 0; l < uint(counts[k].size()); l++) {
      if (counts[k][l] > 0) {
        String_t label;
        label = String_t(NAMEOF_ENUM(AtomicType(k)));
        label += " and ";
        label += String_t(NAMEOF_ENUM(VariableType(l)));
        String_t value;
        value = std::to_string(counts[k][l]);
        emit(label + ": " + value);
      }
    }
  }
  emit("");
}

void Graph::Statistics::gen_roots_and_terminals(uint node_id) {
  tab++;
  uint roots = root_terminal_per_node_type[node_id][0];
  uint terms = root_terminal_per_node_type[node_id][1];
  roots > 0 ? emit("Root nodes: " + std::to_string(roots)) : void(0);
  terms > 0 ? emit("Terminal nodes: " + std::to_string(terms)) : void(0);
  roots == 0 && terms == 0 ? emit("No root or terminal nodes") : void(0);
}

void Graph::Statistics::gen_edge_stats_report() {
  tab = 0;
  emit("Edge statistics:", '#');
  tab++;

  gen_edge_stats_report("incoming", in_edge_histogram);
  gen_edge_stats_report("outgoing", out_edge_histogram);
}

void Graph::Statistics::gen_edge_stats_report(String_t etype, Counts_t counts) {
  emit("Distribution of " + etype + " edges:", '-');
  for (uint i = 0; i < uint(counts.size()); i++) {
    if (counts[i] > 0) {
      String_t label = "Nodes with " + std::to_string(i) + " edges: ";
      String_t value = std::to_string(counts[i]);
      emit(label + value);
    }
  }
  emit("");
}

void Graph::Statistics::emit(String_t output, char banner) {
  if (output != "") {
    emit_tab(tab);
  }
  report << output << std::endl;
  if (banner != '\0') {
    String_t line;
    line.resize(uint(output.size()), banner);
    emit_tab(tab);
    report << line << std::endl;
  }
}

void Graph::Statistics::emit_tab(uint n) {
  for (uint i = 0; i < n; i++) {
    report << "\t";
  }
}

} // namespace graph
} // namespace beanmachine
