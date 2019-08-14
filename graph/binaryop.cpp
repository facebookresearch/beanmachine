// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include <folly/String.h>

#include <beanmachine/graph/binaryop.h>

namespace beanmachine {
namespace oper {

void multiply(graph::Node* node) {
  assert(node->in_nodes.size() > 1);
  const graph::AtomicValue& parent0 = node->in_nodes[0]->value;
  if (parent0.type == graph::AtomicType::REAL) {
    node->value.type = graph::AtomicType::REAL;
    node->value._double = parent0._double;

    for (int i = 1; i < node->in_nodes.size(); i++) {
      const auto& parenti = node->in_nodes[i]->value;
      if (parenti.type != graph::AtomicType::REAL) {
        throw std::runtime_error(folly::stringPrintf(
            "invalid type %d for MULTIPLY operator at node_id %u parent",
            static_cast<int>(parenti.type),
            node->index));
      } else {
        node->value._double *= parenti._double;
      }
    }
  } else if (parent0.type == graph::AtomicType::TENSOR) {
    node->value.type = graph::AtomicType::TENSOR;
    node->value._tensor = parent0._tensor.clone();
    for (int i = 1; i < node->in_nodes.size(); i++) {
      const auto& parenti = node->in_nodes[i]->value;
      if (parenti.type != graph::AtomicType::TENSOR) {
        throw std::runtime_error(folly::stringPrintf(
            "invalid type %d for MULTIPLY operator at node_id %u parent",
            static_cast<int>(parenti.type),
            node->index));
      } else {
        node->value._tensor.mul_(parenti._tensor);
      }
    }
  } else {
    throw std::runtime_error(folly::stringPrintf(
        "invalid type %d for MULTIPLY operator at node_id %u",
        static_cast<int>(parent0.type),
        node->index));
  }
}

void add(graph::Node* node) {
  assert(node->in_nodes.size() > 1);
  const graph::AtomicValue& parent0 = node->in_nodes[0]->value;
  if (parent0.type == graph::AtomicType::REAL) {
    node->value.type = graph::AtomicType::REAL;
    node->value._double = parent0._double;

    for (int i = 1; i < node->in_nodes.size(); i++) {
      const auto& parenti = node->in_nodes[i]->value;
      if (parenti.type != graph::AtomicType::REAL) {
        throw std::runtime_error(folly::stringPrintf(
            "invalid type %d for ADD operator at node_id %u parent",
            static_cast<int>(parenti.type),
            node->index));
      } else {
        node->value._double += parenti._double;
      }
    }
  } else if (parent0.type == graph::AtomicType::TENSOR) {
    node->value.type = graph::AtomicType::TENSOR;
    node->value._tensor = parent0._tensor.clone();
    for (int i = 1; i < node->in_nodes.size(); i++) {
      const auto& parenti = node->in_nodes[i]->value;
      if (parenti.type != graph::AtomicType::TENSOR) {
        throw std::runtime_error(folly::stringPrintf(
            "invalid type %d for ADD operator at node_id %u parent",
            static_cast<int>(parenti.type),
            node->index));
      } else {
        node->value._tensor.add_(parenti._tensor);
      }
    }
  } else {
    throw std::runtime_error(folly::stringPrintf(
        "invalid type %d for ADD operator at node_id %u",
        static_cast<int>(parent0.type),
        node->index));
  }
}

} // namespace oper
} // namespace beanmachine
