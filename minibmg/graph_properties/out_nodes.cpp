/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/graph_properties/out_nodes.h"
#include <exception>
#include <list>
#include <map>

namespace {

using namespace beanmachine::minibmg;

class Out_Nodes_Data {
 public:
  std::map<Nodep, std::list<Nodep>> node_map{};
  std::list<Nodep>& for_node(Nodep node) {
    auto found = node_map.find(node);
    if (found == node_map.end()) {
      throw std::invalid_argument("node not in graph");
    }
    return found->second;
  }
};

class Out_Nodes_Property
    : public Property<Out_Nodes_Property, Graph, Out_Nodes_Data> {
 public:
  Out_Nodes_Data* create(const Graph& g) const override {
    Out_Nodes_Data* data = new Out_Nodes_Data{};
    for (auto node : g) {
      data->node_map[node] = std::list<Nodep>{};
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

const std::list<Nodep>& out_nodes(const Graph& graph, Nodep node) {
  Out_Nodes_Data* data = Out_Nodes_Property::get(graph);
  std::list<Nodep>& result = data->for_node(node);
  return result;
}

} // namespace beanmachine::minibmg
