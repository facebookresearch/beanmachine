/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "beanmachine/minibmg/dedup.h"

namespace beanmachine::minibmg {

// Take a set of root nodes as input, and return a map of (locally) optimized
// nodes, which maps from a node in the transitive closure of the roots to a
// corresponding node in the transitive closure of the optimized graph.
// This is used in the implementation of opt(), but might occasionally be
// useful in this form.
std::unordered_map<Nodep, Nodep> opt_map(std::vector<Nodep> roots);

// Rewrite a data structure by "optimizing" its nodes, applying local
// transformations that are expected to improve runtime required to evaluate it.
template <class T>
T opt(
    const T& data,
    const NodeRewriteAdapter<T>& helper = NodeRewriteAdapter<T>{}) {
  auto roots = helper.find_roots(data);
  auto map = opt_map(roots);
  return helper.rewrite(data, map);
}

} // namespace beanmachine::minibmg
