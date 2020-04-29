// Copyright (c) Facebook, Inc. and its affiliates.

#include "beanmachine/graph/operator/controlop.h"

namespace beanmachine {
namespace oper {

void if_then_else(graph::Node* node) {
  assert(node->in_nodes.size() == 3);
  if (node->in_nodes[0]->value._bool) {
    node->value = node->in_nodes[1]->value;
  } else {
    node->value = node->in_nodes[2]->value;
  }
}

} // namespace oper
} // namespace beanmachine
