/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

class SubGraph : public Graph {
 public:
  explicit SubGraph(Graph& g);
  void add_node_by_id(uint node_id);
  bool has_node(uint node_id);
  void link_copy_node(Node* node, Node* copy_node);
  void move_nodes_from_graph(uint insertion_index = 0, uint insertion_size = 0);

 private:
  Graph& graph;
  std::set<uint> pending_node_ids;
  std::map<Node*, Node*> copy_map;
};

} // namespace graph
} // namespace beanmachine
