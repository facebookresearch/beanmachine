/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include "beanmachine/minibmg/node.h"

namespace beanmachine::minibmg {

// Take a set of root nodes as input, and return a map of deduplicated nodes,
// which maps from a node in the transitive closure of the roots to a
// corresponding node in the transitive closure of the deduplicated graph.
std::unordered_map<Nodep, Nodep> dedup(std::vector<Nodep> roots);

} // namespace beanmachine::minibmg
