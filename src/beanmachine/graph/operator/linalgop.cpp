// Copyright (c) Facebook, Inc. and its affiliates.

#include "beanmachine/graph/operator/linalgop.h"

namespace beanmachine {
namespace oper {

MatrixMultiply::MatrixMultiply(const std::vector<graph::Node*>& in_nodes)
    : Operator(graph::OperatorType::MATRIX_MULTIPLY) {
  if (in_nodes.size() != 2) {
    throw std::invalid_argument(
        "operator MATRIX_MULTIPLY requires two parent nodes");
  }
  graph::ValueType type0 = in_nodes[0]->value.type;
  graph::ValueType type1 = in_nodes[1]->value.type;
  if (type0.variable_type != graph::VariableType::BROADCAST_MATRIX or
      type1.variable_type != graph::VariableType::BROADCAST_MATRIX) {
    throw std::invalid_argument(
        "operator MATRIX_MULTIPLY only supports BROADCAST_MATRIX parents");
  }
  if (type0.atomic_type != type1.atomic_type) {
    throw std::invalid_argument(
        "operator MATRIX_MULTIPLY requires parents have the same AtomicType");
  }
  if (type0.atomic_type != graph::AtomicType::REAL and
      type0.atomic_type != graph::AtomicType::POS_REAL and
      type0.atomic_type != graph::AtomicType::PROBABILITY) {
    throw std::invalid_argument(
        "operator MATRIX_MULTIPLY requires real/pos_real/probability parents");
  }
  if (type0.cols != type1.rows) {
    throw std::invalid_argument(
        "parent nodes have imcompatible dimensions for operator MATRIX_MULTIPLY");
  }

  // Type inferece: R: real, PR: pos real, P: probability
  // R @ R -> R
  // PR @ PR -> PR
  // P @ P -> PR
  graph::ValueType new_type;
  if (type0.rows == 1 and type1.cols == 1) {
    new_type =
        graph::ValueType(graph::VariableType::SCALAR, type0.atomic_type, 0, 0);
  } else {
    new_type = graph::ValueType(
        graph::VariableType::BROADCAST_MATRIX,
        type0.atomic_type,
        type0.rows,
        type1.cols);
  }
  if (type0.atomic_type == graph::AtomicType::PROBABILITY) {
    new_type.atomic_type = graph::AtomicType::POS_REAL;
  }
  value = graph::AtomicValue(new_type);
}

void MatrixMultiply::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 2);
  value._matrix = in_nodes[0]->value._matrix * in_nodes[1]->value._matrix;
  if (value.type.variable_type == graph::VariableType::SCALAR) {
    _to_scalar(value);
  }
}

} // namespace oper
} // namespace beanmachine
