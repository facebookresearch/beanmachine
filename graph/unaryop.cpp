// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include "beanmachine/graph/unaryop.h"

namespace beanmachine {
namespace oper {

void to_real(graph::Node* node) {
  assert(node->in_nodes.size() == 1);
  const graph::AtomicValue& parent = node->in_nodes[0]->value;
  if (parent.type == graph::AtomicType::BOOLEAN) {
    node->value.type = graph::AtomicType::REAL;
    node->value._double = parent._bool ? 1.0 : 0.0;
  } else if (parent.type == graph::AtomicType::REAL) {
    node->value.type = graph::AtomicType::REAL;
    node->value._double = parent._double;
  } else if (parent.type == graph::AtomicType::TENSOR) {
    node->value.type = graph::AtomicType::TENSOR;
    node->value._tensor = parent._tensor.toType(torch::kDouble);
  } else {
    throw std::runtime_error(
      "invalid parent type " + std::to_string(static_cast<int>(parent.type))
      + " for TO_REAL operator at node_id " + std::to_string(node->index));
  }
}

void negate(graph::Node* node) {
  assert(node->in_nodes.size() == 1);
  const graph::AtomicValue& parent = node->in_nodes[0]->value;
  if (parent.type == graph::AtomicType::BOOLEAN) {
    node->value.type = graph::AtomicType::BOOLEAN;
    node->value._bool = parent._bool ? false : true;
  } else if (parent.type == graph::AtomicType::REAL) {
    node->value.type = graph::AtomicType::REAL;
    node->value._double = -parent._double;
  } else if (parent.type == graph::AtomicType::TENSOR) {
    node->value.type = graph::AtomicType::TENSOR;
    node->value._tensor = parent._tensor.neg();
  } else {
    throw std::runtime_error(
      "invalid parent type " + std::to_string(static_cast<int>(parent.type))
      + " for NEGATE operator at node_id " + std::to_string(node->index));
  }
}

void exp(graph::Node* node) {
  assert(node->in_nodes.size() == 1);
  const graph::AtomicValue& parent = node->in_nodes[0]->value;
  if (parent.type == graph::AtomicType::REAL) {
    node->value.type = graph::AtomicType::REAL;
    node->value._double = std::exp(parent._double);
  } else if (parent.type == graph::AtomicType::TENSOR) {
    node->value.type = graph::AtomicType::TENSOR;
    node->value._tensor = parent._tensor.exp();
  } else {
    throw std::runtime_error(
      "invalid parent type " + std::to_string(static_cast<int>(parent.type))
      + " for EXP operator at node_id " + std::to_string(node->index));
  }
}

} // namespace oper
} // namespace beanmachine
