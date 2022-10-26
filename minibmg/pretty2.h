/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <unordered_map>
#include "beanmachine/minibmg/graph2.h"
#include "beanmachine/minibmg/node2.h"

namespace beanmachine::minibmg {

//
// Pretty-printing a set of nodes produces a mapping from each node to the
// string representation of that node.  Because the input nodes may be (directed
// acyclic) graphs rather than trees, in order to avoid producing an exponential
// amout of strings, we produce a "prelude" - a list of assignments of reused
// intermediate results to temporary values.  Tha way each operator in the graph
// to be printed appears exactly once in the output.  For example, if you build
// a tree with this structure
//
//    auto x = observe(...);
//    auto y = x + x;
//    y + y
//
// pretty-printing this last expression naively would result in
//
//    observe(...) + observe(...) + (observe(...) + observe(...))
//
// Instead, we produce the code along with a prelude
//
//    t1 = observe(...)
//    t2 = t1 + t1
//
// and the code is rendered as
//
//    t2 + t2
//
struct Pretty2Result {
  // a set of variable assignments that are prelude to the expressions for the
  // nodes.  These assignments are for shared intermediate results.
  std::vector<std::string> prelude;

  // For each remaining root, the expression for computing it, with identifiers
  // referring to variables declared in the prelude for shared values.
  std::unordered_map<Node2p, std::string> code;
};

// Pretty-print a set of Nodes.  Returns a PrettyResult.
const Pretty2Result pretty_print(std::vector<Node2p> roots);

// Pretty-print a graph into the code that would need to be written using the
// fluid factory to reproduce it.
std::string pretty_print(const Graph2& graph);

} // namespace beanmachine::minibmg
