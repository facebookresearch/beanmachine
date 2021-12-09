/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/operator/linalgop.h"
#include "beanmachine/graph/graph.h"

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
        "parent nodes have incompatible dimensions for MATRIX_MULTIPLY");
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
// TODO[Walid]: The following needs to be modified to actually
// implement the desired functionality

MatrixScale::MatrixScale(const std::vector<graph::Node*>& in_nodes)
    : Operator(graph::OperatorType::MATRIX_SCALE) {
  if (in_nodes.size() != 2) {
    throw std::invalid_argument("MATRIX_SCALE requires two parent nodes");
  }
  graph::ValueType type0 = in_nodes[0]->value.type;
  graph::ValueType type1 = in_nodes[1]->value.type;
  if (type0.variable_type != graph::VariableType::SCALAR and
      type1.variable_type != graph::VariableType::SCALAR) {
    throw std::invalid_argument("MATRIX_SCALE takes one SCALAR parent");
  }
  if (type0.variable_type != graph::VariableType::BROADCAST_MATRIX and
      type1.variable_type != graph::VariableType::BROADCAST_MATRIX) {
    throw std::invalid_argument("MATRIX_SCALE takes one MATRIX parent");
  }
  // TODO[Walid]: Following constraint should go by end of this stack
  if (type0.variable_type != graph::VariableType::SCALAR) {
    throw std::invalid_argument("MATRIX_SCALE takes SCALAR parent first");
  }
  if (type1.variable_type != graph::VariableType::BROADCAST_MATRIX) {
    throw std::invalid_argument(
        "MATRIX_SCALE takes BROADCAST_MATRIX parent first");
  }
  // For the rest, we will follow the same typing rule as for regular
  // multiplication (MULTIPLY)
  // TODO[Walid]: Why not allow Booleans here and in MULTIPLY?
  if (type0 != graph::AtomicType::REAL and
      type0 != graph::AtomicType::POS_REAL and
      type0 != graph::AtomicType::PROBABILITY) {
    throw std::invalid_argument(
        "MATRIX_SCALE requires a real, pos_real or probability parent");
  }

  if (type0.atomic_type != type1.atomic_type) {
    throw std::invalid_argument(
        "MATRIX_SCALE requires both parents have same atomic type");
  }
  graph::ValueType new_type;
  if (type1.rows == 1 and type1.cols == 1) {
    new_type = graph::ValueType(
        graph::VariableType::SCALAR, graph::AtomicType::REAL, 0, 0);
  } else {
    new_type = graph::ValueType(
        graph::VariableType::BROADCAST_MATRIX,
        type1.atomic_type,
        type1.rows,
        type1.cols);
  }
  value = graph::NodeValue(new_type);
}

void MatrixScale::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 2);
  value._matrix = in_nodes[0]->value._double * in_nodes[1]->value._matrix;
  if (value.type.variable_type == graph::VariableType::SCALAR) {
    to_scalar();
  }
}

void MatrixScale::compute_gradients() {
  // TODO[Walid]: This is a naive implementation based simply on the pattern
  // used by ToMatrix:comput_gradients(). A more efficient implementation
  // would incorporate some dynamic programming ideas similar to those used
  // in Multiplication::compute_gradients()
  assert(in_nodes.size() == 2);
  int rows = static_cast<int>(in_nodes[1]->value.type.rows);
  int cols = static_cast<int>(in_nodes[1]->value.type.cols);
  Grad1.resize(rows, cols);
  Grad2.resize(rows, cols);
  // The code currently has a convention that constant values may have
  // zero size Grad1 and Grad2. The purpose of the follow two variables
  // (and their uses) is to accommodate this convention.
  // TODO[Walid]: This seems to be a fairly implicit convention, and it
  // took me a day while working on this diff figure out. It would be good
  // to make this idea more explicit in the code. For example, explicit
  // flags to indicate whether those derivatives are zero (or a proper
  // union type) would help.
  bool hasGrad1 = (in_nodes[1]->Grad1.size() != 0);
  bool hasGrad2 = (in_nodes[1]->Grad2.size() != 0);
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      Grad1(i, j) = in_nodes[0]->grad1 * in_nodes[1]->value._matrix(i, j);
      if (hasGrad1) {
        Grad1(i, j) += in_nodes[0]->value._double * in_nodes[1]->Grad1(i, j);
        Grad2(i, j) = in_nodes[0]->grad2 * in_nodes[1]->Grad1(i, j);
      }
      if (hasGrad2) {
        Grad2(i, j) += in_nodes[0]->grad1 * in_nodes[1]->Grad2(i, j);
      }
    }
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
  if (matrix_index >= static_cast<unsigned long>(matrix.type.rows)) {
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
  value = graph::NodeValue(
      graph::ValueType(type0.variable_type, type0.atomic_type, type0.rows, 1));
}

void ColumnIndex::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 2);
  const graph::NodeValue& matrix = in_nodes[0]->value;
  graph::natural_t matrix_index = in_nodes[1]->value._natural;
  if (matrix_index >= static_cast<unsigned long>(matrix.type.cols)) {
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
