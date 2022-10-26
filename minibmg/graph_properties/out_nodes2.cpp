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

class OutNodesProperty : public Property<
                             OutNodesProperty,
                             Graph2,
                             std::map<Node2p, std::list<Node2p>>> {
 public:
  std::map<Node2p, std::list<Node2p>>* create(const Graph2& g) const override {
    std::map<Node2p, std::list<Node2p>>* data =
        new std::map<Node2p, std::list<Node2p>>{};
    for (auto node : g) {
      (*data)[node] = std::list<Node2p>{};
      for (auto in_node : in_nodes(node)) {
        auto& predecessor_out_set = (*data)[in_node];
        predecessor_out_set.push_back(node);
      }
    }

    return data;
  }
};

} // namespace

namespace beanmachine::minibmg {

const std::list<Node2p>& out_nodes(const Graph2& graph, Node2p node) {
  std::map<Node2p, std::list<Node2p>>& node_map = *OutNodesProperty::get(graph);
  auto found = node_map.find(node);
  if (found == node_map.end()) {
    throw std::invalid_argument("node not in graph");
  }
  return found->second;
}

} // namespace beanmachine::minibmg
