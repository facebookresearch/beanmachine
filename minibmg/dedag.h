/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <concepts>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include "beanmachine/minibmg/node.h"
#include "beanmachine/minibmg/rewrite_adapter.h"
#include "node.h"

namespace beanmachine::minibmg {

// A prelude to a computation, which consists of a sequence of expressions, each
// of which is computed and assigned to a variable.  This permits us to keep
// every expression as a tree (rather than a directed acyclic graph).
using Prelude =
    std::vector<std::pair<std::shared_ptr<const ScalarVariableNode>, Nodep>>;

// The result of "dedagging" a data structure containing nodes.  This rewrites
// the dag so that there are no shared nodes - the result is that every node is
// the root of a tree.  Sharing takes place by computations (subtrees) being
// stored into temporary variables.  Those assignments are indicated by the
// `prelude` part of the result, each entry of which indicates the index of a
// temporary variable, and the expression tree that should be computed into it.
// Previous temporaries are referenced using "Variable" nodes with an index
// corresponding to the int assosicated with the prelude entry.
template <class T>
struct Dedagged {
  Prelude prelude;
  T result;
};

// implemented in dedup.cpp.
std::unordered_map<Nodep, Nodep>
dedag_map(const std::vector<Nodep>& roots, Prelude& prelude, int max_depth);

// Rewrite a data structure by "dedagging" nodes reachable from it, and
// returning a new data structure along with a "prelude" of assignments to
// temporaries that are used to compute intermediate expressions (see
// `Dedagged`).  Performs common subexpression elimination as a side-effect (see
// `dedup`).  Also limits the nesting depth of subtrees to `max_depth`,
// introducing temporaries as needed to enforce that limit.
template <class T, class RewriteAdapter = NodeRewriteAdapter<T>>
requires Rewritable<T, RewriteAdapter>
const Dedagged<T> dedag(const T& data, int max_depth = 15) {
  if (max_depth < 2) {
    throw std::invalid_argument("max_depth must be >= 2");
  }
  RewriteAdapter adapter = RewriteAdapter{};
  std::vector<Nodep> roots = adapter.find_roots(data);
  Prelude prelude{};
  auto map = dedag_map(roots, prelude, max_depth);
  auto result = adapter.rewrite(data, map);
  return Dedagged<T>{std::move(prelude), result};
}

} // namespace beanmachine::minibmg
