/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <utility>

#include "beanmachine/graph/out_nodes_reflexive_transitive_closure.h"
#include "beanmachine/graph/util.h"

namespace beanmachine::graph {

using namespace std;
using namespace util;

OutNodesReflexiveTransitiveClosure::OutNodesReflexiveTransitiveClosure(
    Node* node,
    std::function<pair<bool, bool>(Node*)> prune,
    std::function<pair<bool, bool>(Node*)> abort) {
  // clang-format off
  _success = bfs<Node*, vector<Node*>>(
      node,
      get_out_nodes,
      prune,
      abort,
      push_back_to(_result));
  // clang-format on
}

} // namespace beanmachine::graph
