/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

class MarginalizedGraph : public Graph {
 public:
  explicit MarginalizedGraph(Graph& g);
  void marginalize(uint discrete_node_id);
};

} // namespace graph
} // namespace beanmachine
