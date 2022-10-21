/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <list>
#include <unordered_set>
#include "beanmachine/minibmg/graph2.h"
#include "beanmachine/minibmg/node2.h"

namespace beanmachine::minibmg {

// A map from an observation node to its observed value.
const std::unordered_map<Node2p, double>& observations_by_node(
    const Graph2& graph);

} // namespace beanmachine::minibmg
