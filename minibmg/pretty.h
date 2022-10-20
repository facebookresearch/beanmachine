/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <unordered_map>
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/node.h"

namespace beanmachine::minibmg {

struct Pretty2Result {
  // a set of variable assignments that are prelude to the expressions for the
  // nodes.  These assignments are for shared intermediate results.
  std::vector<std::string> prelude;

  // For each remaining root, the expression for computing it, with identifiers
  // referring to variables declared in the prelude for shared values.
  std::unordered_map<Nodep, std::string> code;
};

// Pretty-print a set of Nodes.  Returns a PrettyResult.
const Pretty2Result pretty_print(std::vector<Nodep> roots);

// Pretty-print a graph into the code that would need to be written using the
// fluid factory to reproduce it.
std::string pretty_print(const Graph& graph);

} // namespace beanmachine::minibmg
