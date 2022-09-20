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
  std::map<const Node*, std::list<const Node*>*> node_map{};
  ~Out_Nodes_Data() {
    for (auto e : node_map) {
      delete e.second;
    }
  }
  std::list<const Node*>& for_node(const Node* node) {
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
    for (auto node : g) {
      data->node_map[node] = new std::list<const Node*>{};
      switch (node->op) {
        case Operator::CONSTANT:
        case Operator::VARIABLE:
          // these nodes do not have inputs.
          break;
        case Operator::QUERY: {
          // query has one input.
          auto query = static_cast<const QueryNode*>(node);
          auto& predecessor_out_set = data->for_node(query->in_node);
          predecessor_out_set.push_back(node);
          break;
        }
        default: {
          // the rest are operator nodes.
          auto opnode = static_cast<const OperatorNode*>(node);
          for (auto in_node : opnode->in_nodes) {
            auto& predecessor_out_set = data->for_node(in_node);
            predecessor_out_set.push_back(node);
          }
          break;
        }
      }
    }

    return data;
  }
};

} // namespace

namespace beanmachine::minibmg {

const std::list<const Node*>& out_nodes(const Graph& graph, const Node* node) {
  Out_Nodes_Data* data = Out_Nodes_Property::get(graph);
  std::list<const Node*>& result = data->for_node(node);
  return result;
}

} // namespace beanmachine::minibmg
