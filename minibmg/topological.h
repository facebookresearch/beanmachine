/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <map>
#include <set>
#include <vector>

namespace beanmachine::minibmg {

// Compute the predecessor count for all nodes reachable from the set of roots
// given.  Parameters:
//
// root_nodes: the set of root nodes from which nodes are identified
//
// successors: a function mapping each node to the nodes that are successors
//
// nodes: a vector that will receive the full set of nodes found reachable
//
// include_roots: whether a root should be counted as a predecessor
template <class T>
std::map<T, unsigned> count_predecessors_internal(
    const std::vector<T>& root_nodes,
    std::function<std::vector<T>(const T&)> successors,
    std::vector<T>& nodes,
    bool include_roots = false) {
  std::map<T, unsigned> predecessor_counts;
  std::vector<T> to_count;
  std::set<T> counted;
  for (const auto& node : root_nodes) {
    to_count.push_back(node);
    if (include_roots) {
      if (!predecessor_counts.contains(node)) {
        predecessor_counts[node] = 1;
      } else {
        predecessor_counts[node] = predecessor_counts[node] + 1;
      }
    }
  }
  std::reverse(to_count.begin(), to_count.end());

  while (!to_count.empty()) {
    auto node = to_count.back();
    to_count.pop_back();
    if (counted.contains(node)) {
      continue;
    }
    counted.insert(node);
    nodes.push_back(node);

    if (!predecessor_counts.contains(node)) {
      predecessor_counts[node] = 0;
    }

    for (const auto& succ : successors(node)) {
      to_count.push_back(succ);
      if (!predecessor_counts.contains(succ)) {
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
bool topological_sort_internal(
    std::map<T, unsigned>& predecessor_counts,
    std::function<std::vector<T>(const T&)> successors,
    std::vector<T>& nodes,
    std::vector<T>& result) {
  // initialize the ready set with those nodes that have no predecessors
  std::vector<T> ready;
  for (auto node : nodes) {
    if (predecessor_counts[node] == 0) {
      ready.push_back(node);
    }
  }

  // process the ready set: output a node, and decrement the predecessor count
  // of its successors
  result = {};
  while (!ready.empty()) {
    auto node = ready.back();
    ready.pop_back();
    result.push_back(node);
    for (const auto& succ : successors(node)) {
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

// Compute the predecessor count for all nodes reachable from the set of roots
// given.  A node 'a' is a predecessor of a node 'b' if node 'b' has node 'a' as
// a successor.  Parameters:
//
// root_nodes: the set of root nodes from which nodes are identified
//
// successors: a function mapping each node to the nodes that are its successors
//
// include_roots: whether or not a root should be counted as a predecessor
template <class T>
std::map<T, unsigned> count_predecessors(
    const std::vector<T>& root_nodes,
    std::function<std::vector<T>(const T&)> successors,
    bool include_roots = false) {
  std::vector<T> ready;
  return count_predecessors_internal<T>(
      root_nodes, successors, ready, include_roots);
}

// Perform a topological sort of the nodes in the directed acyclic graph imposed
// by the set of root nodes and the successor relation.  Returns true if the
// topological sort succeeds (no cycle was found).  Returns the topologically
// sorted result in the `result` parameter.
template <class T>
bool topological_sort(
    const std::vector<T>& root_nodes,
    std::function<std::vector<T>(const T&)> successors,
    std::vector<T>& result) {
  std::vector<T> ready;
  // count the predecessors of each node.
  std::map<T, unsigned> predecessor_counts =
      count_predecessors_internal<T>(root_nodes, successors, ready);
  return topological_sort_internal<T>(
      predecessor_counts, successors, ready, result);
}

} // namespace beanmachine::minibmg
