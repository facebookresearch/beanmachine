/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <list>
#include <unordered_set>
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/node.h"

namespace beanmachine::minibmg {

const std::vector<Nodep>& unobserved_samples(const Graph& graph);

} // namespace beanmachine::minibmg
