/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/marginalization/copy_node.h"

namespace beanmachine {
namespace graph {

CopyNode::CopyNode(Node* node_to_copy) : Node(NodeType::COPY) {
  this->node_to_copy = node_to_copy;
  value = node_to_copy->value;
}

void CopyNode::eval(std::mt19937& /*gen*/) {
  value = node_to_copy->value;
}

double CopyNode::log_prob() const {
  return node_to_copy->log_prob();
}
} // namespace graph
} // namespace beanmachine
