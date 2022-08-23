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
  std::set<uint> get_node_ids();
  void add_node_by_id(uint node_id);
  bool has_node(uint node_id);
  void move_nodes_from_graph_and_reindex();

 private:
  Graph& graph;
  std::set<uint> pending_node_ids;
  std::map<Node*, Node*> copy_map;
};

} // namespace graph
} // namespace beanmachine
