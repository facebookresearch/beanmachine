/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_set>
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/node.h"

namespace beanmachine::minibmg {

// A map from an observation node to its observed value.
const std::unordered_map<Nodep, double>& observations_by_node(
    const Graph& graph);

} // namespace beanmachine::minibmg
