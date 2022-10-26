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

class OutNodesProperty : public Property<
                             OutNodesProperty,
                             Graph,
                             std::map<Nodep, std::list<Nodep>>> {
 public:
  std::map<Nodep, std::list<Nodep>>* create(const Graph& g) const override {
    std::map<Nodep, std::list<Nodep>>* data =
        new std::map<Nodep, std::list<Nodep>>{};
    for (auto node : g) {
      (*data)[node] = std::list<Nodep>{};
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

const std::list<Nodep>& out_nodes(const Graph& graph, Nodep node) {
  std::map<Nodep, std::list<Nodep>>& node_map = *OutNodesProperty::get(graph);
  auto found = node_map.find(node);
  if (found == node_map.end()) {
    throw std::invalid_argument("node not in graph");
  }
  return found->second;
}

} // namespace beanmachine::minibmg
