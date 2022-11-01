/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <unordered_set>
#include <vector>

#include "beanmachine/graph/graph.h"

namespace beanmachine::graph {

/*
A class representing the out-nodes reflexive transitive closure of a node in a
graph, that is, the set of itself and its descendants.

The class receives two optional functions for controlling the search:

- the 'abort' function returns a pair of booleans. The first boolean indicates
whether the search is to be aborted, whereas the second indicates whether the
node causing the abort should be excluded from the result.

- the 'prune' function returns a pair of booleans. The first boolean indicates
whether the search is to be pruned, whether the second indicates whether the
node causing the pruning should be excluded from the result.
*/
class OutNodesReflexiveTransitiveClosure {
 public:
  explicit OutNodesReflexiveTransitiveClosure(
      Node* node,
      std::function<std::pair<bool, bool>(Node*)> prune =
          [](Node*) {
            return std::pair<bool, bool>{false, false};
          },
      std::function<std::pair<bool, bool>(Node*)> abort =
          [](Node*) { return std::pair<bool, bool>(false, false); });

  const std::vector<Node*> get_result() const {
    return _result;
  }

  bool aborted() const {
    return not _success;
  }

  bool success() const {
    return _success;
  }

 private:
  bool _success;
  std::vector<Node*> _result;
};

} // namespace beanmachine::graph
