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

 private:
  Graph& graph;
};

} // namespace graph
} // namespace beanmachine
