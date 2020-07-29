// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include "beanmachine/graph/operator/binaryop.h"

namespace beanmachine {
namespace oper {

void multiply(graph::Node* node) {
  assert(node->in_nodes.size() > 1);
  const graph::AtomicValue& parent0 = node->in_nodes[0]->value;
  if (parent0.type == graph::AtomicType::REAL or
      parent0.type == graph::AtomicType::POS_REAL or
      parent0.type == graph::AtomicType::PROBABILITY) {
    node->value._double = parent0._double;

    for (uint i = 1; i < node->in_nodes.size(); i++) {
      const auto& parenti = node->in_nodes[i]->value;
      node->value._double *= parenti._double;
    }
  } else if (parent0.type == graph::AtomicType::TENSOR) {
    node->value._tensor = parent0._tensor.clone();

    for (uint i = 1; i < node->in_nodes.size(); i++) {
      const auto& parenti = node->in_nodes[i]->value;
      node->value._tensor.mul_(parenti._tensor);
    }
  } else {
    throw std::runtime_error(
        "invalid type " + parent0.type.to_string() +
        " for MULTIPLY operator at node_id " + std::to_string(node->index));
  }
}

void add(graph::Node* node) {
  assert(node->in_nodes.size() > 1);
  const graph::AtomicValue& parent0 = node->in_nodes[0]->value;
  if (parent0.type == graph::AtomicType::REAL or
      parent0.type == graph::AtomicType::POS_REAL) {
    node->value._double = parent0._double;

    for (uint i = 1; i < node->in_nodes.size(); i++) {
      const auto& parenti = node->in_nodes[i]->value;
      node->value._double += parenti._double;
    }
  } else if (parent0.type == graph::AtomicType::TENSOR) {
    node->value._tensor = parent0._tensor.clone();

    for (uint i = 1; i < node->in_nodes.size(); i++) {
      const auto& parenti = node->in_nodes[i]->value;
      node->value._tensor.add_(parenti._tensor);
    }
  } else {
    throw std::runtime_error(
        "invalid type " + parent0.type.to_string() +
        " for ADD operator at node_id " + std::to_string(node->index));
  }
}

void logsumexp(graph::Node* node) {
  assert(node->in_nodes.size() > 1);
  const graph::AtomicValue& parent0 = node->in_nodes[0]->value;
  if (parent0.type == graph::AtomicType::REAL or
      parent0.type == graph::AtomicType::POS_REAL) {
    double max_val = parent0._double;
    for (uint i = 1; i < node->in_nodes.size(); i++) {
      const auto& parenti = node->in_nodes[i]->value;
      if (parenti._double > max_val) {
        max_val = parenti._double;
      }
    }
    double expsum = 0.0;
    for (const auto parent : node->in_nodes) {
      expsum += std::exp(parent->value._double - max_val);
    }
    node->value._double = std::log(expsum) + max_val;
  } else {
    throw std::runtime_error(
        "invalid type " + parent0.type.to_string() +
        " for LOGSUMEXP operator at node_id " + std::to_string(node->index));
  }
}

} // namespace oper
} // namespace beanmachine
