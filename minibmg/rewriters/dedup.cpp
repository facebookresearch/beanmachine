/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/rewriters/dedup.h"
#include <map>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include "beanmachine/minibmg/node.h"
#include "beanmachine/minibmg/rewriters/dedag.h"
#include "beanmachine/minibmg/rewriters/update_children.h"
#include "beanmachine/minibmg/topological.h"

namespace beanmachine::minibmg {

// Take a set of root nodes as input, and return a map of deduplicated nodes,
// which maps from a node in the transitive closure of the input to a
// corresponding node in the transitive closure of the deduplicated graph.
std::unordered_map<Nodep, Nodep> dedup_map(const std::vector<Nodep>& roots) {
  // a value-based, map, which treats semantically identical nodes as the same.
  NodeNodeValueMap map;

  // We also build a map that uses object (pointer) identity to find elements,
  // so that operations in clients are not using recursive equality operations.
  std::unordered_map<Nodep, Nodep> identity_map;

  std::vector<Nodep> sorted;
  if (!topological_sort<Nodep>(
          {roots.begin(), roots.end()}, in_nodes, sorted)) {
    throw std::invalid_argument("graph has a cycle");
  }
  std::reverse(sorted.begin(), sorted.end());

  // Compute a replacement for each node.
  for (auto& node : sorted) {
    auto found = map.find(node);
    if (found != map.end()) {
      auto mapping = found->second;
      identity_map.insert({node, mapping});
    } else {
      auto rewritten = update_children(node, identity_map);
      map.insert(node, rewritten);
      identity_map.insert({node, rewritten});
    }
  }

  return identity_map;
}

} // namespace beanmachine::minibmg
