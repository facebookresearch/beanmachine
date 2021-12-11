/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace proposer {

/*
Sets a default (or uniformly sampled) value for the specified node.
:param gen: A random number generator
:param node: The desired node the value of which is to be set
*/
void default_initializer(std::mt19937& gen, graph::Node* node);

} // namespace proposer
} // namespace beanmachine
