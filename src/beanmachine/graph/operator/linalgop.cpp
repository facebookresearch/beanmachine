// Copyright (c) Facebook, Inc. and its affiliates.

#include "beanmachine/graph/operator/linalgop.h"

/*
A MACRO that checks the atomic_type of a node to make sure the underlying
data is stored in _matrix.
*/
#define CHECK_TYPE_DOUBLE(atomic_type)                                            \
  switch (atomic_type) {                                                          \
    case graph::AtomicType::REAL:                                                 \
    case graph::AtomicType::POS_REAL:                                             \
    case graph::AtomicType::NEG_REAL:                                             \
    case graph::AtomicType::PROBABILITY:                                          \
      break;                                                                      \
    default:                                                                      \
      throw std::invalid_argument(                                                \
          "MATRIX_MULTIPLY requires real/pos_real/neg_real/probability parents"); \
  }

namespace beanmachine {
namespace oper {

MatrixMultiply::MatrixMultiply(const std::vector<graph::Node*>& in_nodes)
    : Operator(graph::OperatorType::MATRIX_MULTIPLY) {
  if (in_nodes.size() != 2) {
    throw std::invalid_argument("MATRIX_MULTIPLY requires two parent nodes");
  }
  graph::ValueType type0 = in_nodes[0]->value.type;
  graph::ValueType type1 = in_nodes[1]->value.type;
  if (type0.variable_type == graph::VariableType::SCALAR or
      type1.variable_type == graph::VariableType::SCALAR) {
    throw std::invalid_argument("MATRIX_MULTIPLY cannot have SCALAR parents");
  }
  CHECK_TYPE_DOUBLE(type0.atomic_type)
  CHECK_TYPE_DOUBLE(type1.atomic_type)
  if (type0.cols != type1.rows) {
    throw std::invalid_argument(
        "parent nodes have imcompatible dimensions for MATRIX_MULTIPLY");
  }
  // AtomicType inference is not rigorous, we assume
  // (R or pos_R or neg_R or Prob) @ (R or pos_R or neg_R or Prob) -> R
  graph::ValueType new_type;
  if (type0.rows == 1 and type1.cols == 1) {
    new_type = graph::ValueType(
        graph::VariableType::SCALAR, graph::AtomicType::REAL, 0, 0);
  } else {
    new_type = graph::ValueType(
        graph::VariableType::BROADCAST_MATRIX,
        graph::AtomicType::REAL,
        type0.rows,
        type1.cols);
  }
  value = graph::NodeValue(new_type);
}

void MatrixMultiply::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 2);
  value._matrix = in_nodes[0]->value._matrix * in_nodes[1]->value._matrix;
  if (value.type.variable_type == graph::VariableType::SCALAR) {
    to_scalar();
  }
}

} // namespace oper
} // namespace beanmachine
