/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <concepts>
#include <list>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include "beanmachine/minibmg/node.h"
#include "beanmachine/minibmg/rewrite_adapter.h"

namespace beanmachine::minibmg {

// Take a set of root nodes as input, and return a map of deduplicated nodes,
// which maps from a node in the transitive closure of the roots to a
// corresponding node in the transitive closure of the deduplicated graph.
// A deduplicated graph is one in which there is only one (shared) copy of
// equivalent expressions (that is, every pair of distinct nodes in the
// resulting data structure are semantically different). This is used in the
// implementation of dedup(), but might occasionally be useful to clients in
// this form.
std::unordered_map<Nodep, Nodep> dedup_map(std::vector<Nodep> roots);

// Rewrite a data structure by "deduplicating" nodes reachable from it, and
// returning a new data structure.  This is also known as common subexpression
// elimination.  The nodes in the resulting data structure will have only one
// (shared) copy of equivalent expressions (that is, every pair of distinct
// nodes in the resulting data structure are semantically different). The
// programmer must either specialize DedupAdapter<T> or provide a type to be
// used in its place.  If the input has a tree of nodes, the result may contain
// a DAG (directed acyclic graph) of nodes that is no longer a tree due to
// shared (semantically equivalent) subexpressions.
template <class T, class RewriteAdapter = NodeRewriteAdapter<T>>
requires Rewritable<T, RewriteAdapter> T
dedup(const T& data, std::unordered_map<Nodep, Nodep>* ddmap = nullptr) {
  RewriteAdapter adapter = RewriteAdapter{};
  auto roots = adapter.find_roots(data);
  auto map = dedup_map(roots);
  if (ddmap != nullptr) {
    ddmap->insert(map.begin(), map.end());
  }
  return adapter.rewrite(data, map);
}

} // namespace beanmachine::minibmg
