/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_set>
#include <vector>
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/node.h"

namespace beanmachine::minibmg {

// return the set of nodes that have the given node as an input in the given
// graph.
const std::vector<Nodep>& out_nodes(const Graph& graph, Nodep node);

} // namespace beanmachine::minibmg
