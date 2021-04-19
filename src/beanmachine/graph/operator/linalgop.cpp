// Copyright (c) Facebook, Inc. and its affiliates.

#include "beanmachine/graph/operator/linalgop.h"

/*
A MACRO that checks the atomic_type of a node to make sure the underlying
data is stored in _matrix.
*/
#define CHECK_TYPE_DOUBLE(atomic_type, operator)                           \
  switch (atomic_type) {                                                   \
    case graph::AtomicType::REAL:                                          \
    case graph::AtomicType::POS_REAL:                                      \
    case graph::AtomicType::NEG_REAL:                                      \
    case graph::AtomicType::PROBABILITY:                                   \
      break;                                                               \
    default:                                                               \
      throw std::invalid_argument(                                         \
          operator" requires real/pos_real/neg_real/probability parents"); \
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
  CHECK_TYPE_DOUBLE(type0.atomic_type, "MATRIX_MULTIPLY")
  CHECK_TYPE_DOUBLE(type1.atomic_type, "MATRIX_MULTIPLY")
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

Index::Index(const std::vector<graph::Node*>& in_nodes)
    : Operator(graph::OperatorType::INDEX) {
  if (in_nodes.size() != 2) {
    throw std::invalid_argument("INDEX requires two parent nodes");
  }
  graph::ValueType type0 = in_nodes[0]->value.type;
  if ((type0.cols != 1) or
      !((type0.variable_type == graph::VariableType::BROADCAST_MATRIX) or
        (type0.variable_type == graph::VariableType::COL_SIMPLEX_MATRIX))) {
    throw std::invalid_argument(
        "the first parent of INDEX must be a MATRIX with one column");
  }
  graph::ValueType type1 = in_nodes[1]->value.type;
  if (type1 != graph::AtomicType::NATURAL) {
    // TODO: change type to ranged natural
    throw std::invalid_argument(
        "the second parent of INDEX must be NATURAL number");
  }
  value = graph::NodeValue(graph::ValueType(type0.atomic_type));
}

void Index::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 2);
  const graph::NodeValue& matrix = in_nodes[0]->value;
  graph::natural_t matrix_index = in_nodes[1]->value._natural;
  if (matrix_index >= matrix.type.rows) {
    throw std::runtime_error(
        "invalid index for INDEX operator at node_id " + std::to_string(index));
  }
  graph::AtomicType matrix_type = matrix.type.atomic_type;
  if (matrix_type == graph::AtomicType::BOOLEAN) {
    value._bool = matrix._bmatrix(matrix_index);
  } else if (
      matrix_type == graph::AtomicType::REAL or
      matrix_type == graph::AtomicType::POS_REAL or
      matrix_type == graph::AtomicType::NEG_REAL or
      matrix_type == graph::AtomicType::PROBABILITY) {
    value._double = matrix._matrix(matrix_index);
  } else if (matrix_type == graph::AtomicType::NATURAL) {
    value._natural = matrix._nmatrix(matrix_index);
  } else {
    throw std::runtime_error(
        "invalid parent type " + matrix.type.to_string() +
        " for INDEX operator at node_id " + std::to_string(index));
  }
}

ColumnIndex::ColumnIndex(const std::vector<graph::Node*>& in_nodes)
    : Operator(graph::OperatorType::COLUMN_INDEX) {
  if (in_nodes.size() != 2) {
    throw std::invalid_argument("COLUMN_INDEX requires two parent nodes");
  }
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (!((type0.variable_type == graph::VariableType::BROADCAST_MATRIX) or
        (type0.variable_type == graph::VariableType::COL_SIMPLEX_MATRIX))) {
    throw std::invalid_argument(
        "the first parent of COLUMN_INDEX must be a MATRIX");
  }
  graph::ValueType type1 = in_nodes[1]->value.type;
  if (type1 != graph::AtomicType::NATURAL) {
    // TODO: change type to ranged natural
    throw std::invalid_argument(
        "the second parent of COLUMN_INDEX must be NATURAL number");
  }
  value = graph::NodeValue(graph::ValueType(
      graph::VariableType::BROADCAST_MATRIX, type0.atomic_type, type0.rows, 1));
}

void ColumnIndex::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 2);
  const graph::NodeValue& matrix = in_nodes[0]->value;
  graph::natural_t matrix_index = in_nodes[1]->value._natural;
  if (matrix_index >= matrix.type.cols) {
    throw std::runtime_error(
        "invalid index for COLUMN_INDEX at node_id " + std::to_string(index));
  }
  graph::AtomicType matrix_type = matrix.type.atomic_type;
  if (matrix_type == graph::AtomicType::BOOLEAN) {
    value._bmatrix = matrix._bmatrix.col(matrix_index);
  } else if (
      matrix_type == graph::AtomicType::REAL or
      matrix_type == graph::AtomicType::POS_REAL or
      matrix_type == graph::AtomicType::NEG_REAL or
      matrix_type == graph::AtomicType::PROBABILITY) {
    value._matrix = matrix._matrix.col(matrix_index);
  } else if (matrix_type == graph::AtomicType::NATURAL) {
    value._nmatrix = matrix._nmatrix.col(matrix_index);
  } else {
    throw std::runtime_error(
        "invalid parent type " + matrix.type.to_string() +
        " for COLUMN_INDEX operator at node_id " + std::to_string(index));
  }
}

BroadcastAdd::BroadcastAdd(const std::vector<graph::Node*>& in_nodes)
    : Operator(graph::OperatorType::BROADCAST_ADD) {
  if (in_nodes.size() != 2) {
    throw std::invalid_argument("BROADCAST_ADD requires two parent nodes");
  }
  auto type0 = in_nodes[0]->value.type;
  if (type0.variable_type != graph::VariableType::SCALAR) {
    throw std::invalid_argument(
        "the first parent of BROADCAST_ADD must be a SCALAR");
  }
  auto type1 = in_nodes[1]->value.type;
  if (!((type1.variable_type == graph::VariableType::BROADCAST_MATRIX) or
        (type1.variable_type == graph::VariableType::COL_SIMPLEX_MATRIX))) {
    throw std::invalid_argument(
        "the second parent of BROADCAST_ADD must be a MATRIX");
  }
  CHECK_TYPE_DOUBLE(type0.atomic_type, "BROADCAST_ADD")
  CHECK_TYPE_DOUBLE(type1.atomic_type, "BROADCAST_ADD")
  value = graph::NodeValue(graph::ValueType(
      graph::VariableType::BROADCAST_MATRIX,
      graph::AtomicType::REAL,
      type1.rows,
      type1.cols));
}

void BroadcastAdd::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 2);
  value._matrix =
      in_nodes[0]->value._double + in_nodes[1]->value._matrix.array();
}

} // namespace oper
} // namespace beanmachine
