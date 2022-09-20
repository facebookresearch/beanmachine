/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/out_nodes.h"
#include <exception>
#include <list>
#include <map>
#include <unordered_set>
#include "beanmachine/minibmg/minibmg.h"

namespace {

using namespace beanmachine::minibmg;

class Out_Nodes_Data {
 public:
  std::map<const Node*, std::unordered_set<NodeId>*> id_map{};
  std::map<const Node*, std::list<const Node*>*> node_map{};
  ~Out_Nodes_Data() {
    for (auto e : id_map) {
      delete e.second;
    }
    for (auto e : node_map) {
      delete e.second;
    }
  }
  std::unordered_set<NodeId>& for_ids(const Node* node) {
    auto found = id_map.find(node);
    if (found == id_map.end()) {
      throw std::invalid_argument("node not in graph");
    }
    return *found->second;
  }
  std::list<const Node*>& for_nodes(const Node* node) {
    auto found = node_map.find(node);
    if (found == node_map.end()) {
      throw std::invalid_argument("node not in graph");
    }
    return *found->second;
  }
};

class Out_Nodes_Property
    : public Property<Out_Nodes_Property, Graph, Out_Nodes_Data> {
 public:
  Out_Nodes_Data* create(const Graph& g) const override {
    Out_Nodes_Data* data = new Out_Nodes_Data{};
    // create the version using NodeId values
    for (auto node : g) {
      data->id_map[node] = new std::unordered_set<NodeId>{};
      switch (node->op) {
        case Operator::CONSTANT:
        case Operator::VARIABLE:
          // these nodes do not have inputs.
          break;
        case Operator::QUERY: {
          // query has one input.
          auto query = static_cast<const QueryNode*>(node);
          auto predecessor_out_set = data->id_map[query->in_node];
          predecessor_out_set->insert(node->sequence);
          break;
        }
        default: {
          // the rest are operator nodes.
          auto opnode = static_cast<const OperatorNode*>(node);
          for (auto in_node : opnode->in_nodes) {
            auto predecessor_out_set = data->id_map[in_node];
            predecessor_out_set->insert(node->sequence);
          }
          break;
        }
      }
    }

    // create the version using const Node* values
    for (auto node : g) {
      auto new_list = new std::list<const Node*>{};
      auto& uset = *data->id_map[node];
      for (auto u : uset) {
        new_list->push_back(g[u]);
      }
      data->node_map[node] = new_list;
    }

    return data;
  }
};

} // namespace

namespace beanmachine::minibmg {

const std::unordered_set<NodeId>& out_nodes(const Graph& graph, NodeId node) {
  const Node* n = graph[node];
  if (n == nullptr) {
    throw std::invalid_argument("node not in graph");
  }
  Out_Nodes_Data* data = Out_Nodes_Property::get(graph);
  std::unordered_set<NodeId>& result = data->for_ids(graph[node]);
  return result;
}

const std::list<const Node*>& out_nodes(const Graph& graph, const Node* node) {
  Out_Nodes_Data* data = Out_Nodes_Property::get(graph);
  std::list<const Node*>& result = data->for_nodes(node);
  return result;
}

} // namespace beanmachine::minibmg
