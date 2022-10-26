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
#include "beanmachine/minibmg/minibmg.h"

namespace {

using namespace beanmachine::minibmg;

class OutNodesData {
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

class OutNodesProperty
    : public Property<OutNodesProperty, Graph, OutNodesData> {
 public:
  OutNodesData* create(const Graph& g) const override {
    OutNodesData* data = new OutNodesData{};
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
  OutNodesData* data = OutNodesProperty::get(graph);
  std::list<Nodep>& result = data->for_node(node);
  return result;
}

} // namespace beanmachine::minibmg
