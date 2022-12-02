/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "beanmachine/minibmg/node.h"

namespace beanmachine::minibmg {

// If the node represents an operation (not a distribution) all of whose inputs
// are constants, return a constant node representing the resulting value.
// Otherwise returns the original node.  Note that this does not act
// recursively; input nodes should be folded before calling this.
Nodep constant_fold(Nodep node);
ScalarNodep constant_fold(ScalarNodep node);

} // namespace beanmachine::minibmg
