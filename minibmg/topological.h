/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <list>
#include <map>
#include <set>

namespace beanmachine::minibmg {

// Compute the predecessor count for all nodes reachable from the set of roots
// given.
template <class T>
std::map<T, uint> count_predecessors(
    const std::list<T>& root_nodes,
    std::function<std::vector<T>(const T&)> successors) {
  std::map<T, uint> predecessor_counts;
  std::list<T> to_count;
  std::set<T> counted;
  for (auto node : root_nodes) {
    to_count.push_back(node);
  }

  while (!to_count.empty()) {
    auto node = to_count.back();
    to_count.pop_back();
    if (counted.contains(node)) {
      continue;
    }
    counted.insert(node);

    if (predecessor_counts.find(node) == predecessor_counts.end()) {
      predecessor_counts[node] = 0;
    }

    for (auto succ : successors(node)) {
      to_count.push_back(succ);
      auto found = predecessor_counts.find(succ);
      if (found == predecessor_counts.end()) {
        predecessor_counts[succ] = 1;
      } else {
        predecessor_counts[succ] = predecessor_counts[succ] + 1;
      }
    }
  }

  return predecessor_counts;
}

// Perform a topological sort of the nodes in the directed acyclic graph imposed
// by a set of root nodes and the successor relation.  The `predecessor_counts`
// parameter contains a map with all of the nodes reachable from the roots as
// keys, and the number of incoming edges as values.  Returns true if the
// topological sort succeeds (no cycle was found).  Returns the topologically
// sorted result in the `result` parameter.  Note: clears `predecessor_counts`.
template <class T>
bool topological_sort(
    std::map<T, uint>& predecessor_counts,
    std::function<std::vector<T>(const T&)> successors,
    std::vector<T>& result) {
  // initialize the ready set with those nodes that have no predecessors
  std::list<T> ready;
  for (auto node_and_count : predecessor_counts) {
    if (node_and_count.second == 0) {
      ready.push_back(node_and_count.first);
    }
  }

  // process the ready set: output a node, and decrement the predecessor count
  // of its successors
  result = {};
  while (!ready.empty()) {
    auto node = ready.back();
    ready.pop_back();
    result.push_back(node);
    for (auto succ : successors(node)) {
      auto count = predecessor_counts[succ];
      count--;
      predecessor_counts[succ] = count;
      if (count == 0) {
        ready.push_back(succ);
      }
    }
  }

  // at this point all the nodes should have been output, otherwise there was a
  // cycle.  We return true when we succeed (no cycle).
  return predecessor_counts.size() == result.size();
}

// Perform a topological sort of the nodes in the directed acyclic graph imposed
// by the set of root nodes and the successor relation.  Returns true if the
// topological sort succeeds (no cycle was found).  Returns the topologically
// sorted result in the `result` parameter.
template <class T>
bool topological_sort(
    const std::list<T>& root_nodes,
    std::function<std::vector<T>(const T&)> successors,
    std::vector<T>& result) {
  // count the predecessors of each node.
  std::map<T, uint> predecessor_counts =
      count_predecessors(root_nodes, successors);
  return topological_sort(predecessor_counts, successors, result);
}

} // namespace beanmachine::minibmg
