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
  void move_nodes_from_graph();

 private:
  Graph& graph;
  std::set<uint> pending_node_ids;
};

} // namespace graph
} // namespace beanmachine
