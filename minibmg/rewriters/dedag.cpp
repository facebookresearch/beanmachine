/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/rewriters/dedag.h"
#include <map>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include "beanmachine/minibmg/node.h"
#include "beanmachine/minibmg/rewriters/update_children.h"
#include "beanmachine/minibmg/topological.h"

namespace beanmachine::minibmg {

std::unordered_map<Nodep, Nodep> dedag_map(
    const std::vector<Nodep>& roots,
    std::vector<std::pair<std::shared_ptr<const ScalarVariableNode>, Nodep>>&
        prelude,
    int max_depth) {
  // a value-based, map, which treats semantically identical nodes as the same.
  int next_variable = 0;
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

  std::unordered_map<Nodep, int> depth_map{};

  std::vector<Nodep> unused;
  std::map<Nodep, unsigned> predecessor_count =
      count_predecessors_internal<Nodep>(
          roots, in_nodes, unused, /* include_roots = */ true);

  // Compute a replacement for each node.
  for (auto& node : sorted) {
    auto found = map.find(node);
    if (found != map.end()) {
      auto mapping = found->second;
      identity_map.insert({node, mapping});
    } else {
      auto rewritten = update_children(node, identity_map);
      int depth = 1;
      for (auto in_node : in_nodes(rewritten)) {
        depth = std::max(depth, 1 + depth_map.at(in_node));
      }
      if (depth >= max_depth || (depth > 1 && predecessor_count.at(node) > 1)) {
        depth_map.insert({rewritten, depth});
        int this_variable = next_variable++;
        auto variable_name = fmt::format("_temp{}", this_variable);
        // temp indices are negative
        auto variable_index = ~this_variable;
        auto variable = std::make_shared<const ScalarVariableNode>(
            variable_name, variable_index);
        prelude.push_back(std::pair(variable, rewritten));
        rewritten = variable;
        depth = 1;
      }
      map.insert(node, rewritten);
      depth_map.insert({rewritten, depth});
      identity_map.insert({node, rewritten});
    }
  }

  return identity_map;
}

} // namespace beanmachine::minibmg
