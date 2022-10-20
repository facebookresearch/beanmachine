/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/graph_properties/out_nodes2.h"
#include <exception>
#include <list>
#include <map>

namespace {

using namespace beanmachine::minibmg;

class Out_Nodes_Data {
 public:
  std::map<Node2p, std::list<Node2p>> node_map{};
  std::list<Node2p>& for_node(Node2p node) {
    auto found = node_map.find(node);
    if (found == node_map.end()) {
      throw std::invalid_argument("node not in graph");
    }
    return found->second;
  }
};

class Out_Nodes_Property
    : public Property<Out_Nodes_Property, Graph2, Out_Nodes_Data> {
 public:
  Out_Nodes_Data* create(const Graph2& g) const override {
    Out_Nodes_Data* data = new Out_Nodes_Data{};
    for (auto node : g) {
      data->node_map[node] = std::list<Node2p>{};
      for (auto in_node : in_nodes(node)) {
        auto& predecessor_out_set = data->for_node(in_node);
        predecessor_out_set.push_back(node);
      }
    }

    return data;
  }
};

} // namespace

namespace beanmachine::minibmg {

const std::list<Node2p>& out_nodes(const Graph2& graph, Node2p node) {
  Out_Nodes_Data* data = Out_Nodes_Property::get(graph);
  std::list<Node2p>& result = data->for_node(node);
  return result;
}

} // namespace beanmachine::minibmg
