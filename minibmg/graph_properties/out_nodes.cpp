/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/graph_properties/out_nodes.h"
#include <exception>
#include <map>

namespace {

using namespace beanmachine::minibmg;

class OutNodesProperty : public Property<
                             OutNodesProperty,
                             Graph,
                             std::map<Nodep, std::vector<Nodep>>> {
 public:
  std::map<Nodep, std::vector<Nodep>>* create(const Graph& g) const override {
    std::map<Nodep, std::vector<Nodep>>* data =
        new std::map<Nodep, std::vector<Nodep>>{};
    for (auto node : g) {
      (*data)[node] = std::vector<Nodep>{};
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

const std::vector<Nodep>& out_nodes(const Graph& graph, Nodep node) {
  std::map<Nodep, std::vector<Nodep>>& node_map = *OutNodesProperty::get(graph);
  auto found = node_map.find(node);
  if (found == node_map.end()) {
    throw std::invalid_argument("node not in graph");
  }
  return found->second;
}

} // namespace beanmachine::minibmg
