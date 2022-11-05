/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>
#include "beanmachine/minibmg/rewriters/rewrite_adapter.h"

namespace beanmachine::minibmg {

// Update a node by replacing its immediate in_nodes according to the given map.
// Note that this does not act recursively, and it requires that each in_node
// appears as a key in the map.
Nodep update_children(
    const Nodep& node,
    const std::unordered_map<Nodep, Nodep>& map);
ScalarNodep update_children(
    const ScalarNodep& node,
    const std::unordered_map<Nodep, Nodep>& map);

} // namespace beanmachine::minibmg
