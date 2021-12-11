/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include "beanmachine/graph/operator/multiaryop.h"

namespace beanmachine {
namespace oper {

Add::Add(const std::vector<graph::Node*>& in_nodes)
    : MultiaryOperator(graph::OperatorType::ADD, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::REAL and
      type0 != graph::AtomicType::POS_REAL and
      type0 != graph::AtomicType::NEG_REAL) {
    throw std::invalid_argument(
        "operator ADD requires a real, pos_real or neg_real parent");
  }
  value = graph::NodeValue(type0);
}

void Add::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() > 1);
  const graph::NodeValue& parent0 = in_nodes[0]->value;
  if (parent0.type == graph::AtomicType::REAL or
      parent0.type == graph::AtomicType::POS_REAL or
      parent0.type == graph::AtomicType::NEG_REAL) {
    value._double = parent0._double;

    for (uint i = 1; i < static_cast<uint>(in_nodes.size()); i++) {
      const auto& parenti = in_nodes[i]->value;
      value._double += parenti._double;
    }
  } else {
    throw std::runtime_error(
        "invalid type " + parent0.type.to_string() +
        " for ADD operator at node_id " + std::to_string(index));
  }
}

// TODO[Walid]: Why not allow Booleans here and in MATRIX_SCALE?
Multiply::Multiply(const std::vector<graph::Node*>& in_nodes)
    : MultiaryOperator(graph::OperatorType::MULTIPLY, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::REAL and
      type0 != graph::AtomicType::POS_REAL and
      type0 != graph::AtomicType::PROBABILITY) {
    throw std::invalid_argument(
        "operator MUTIPLY requires a real, pos_real or probability parent");
  }
  value = graph::NodeValue(type0);
}

void Multiply::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() > 1);
  const graph::NodeValue& parent0 = in_nodes[0]->value;
  if (parent0.type == graph::AtomicType::REAL or
      parent0.type == graph::AtomicType::POS_REAL or
      parent0.type == graph::AtomicType::PROBABILITY) {
    value._double = parent0._double;

    for (uint i = 1; i < static_cast<uint>(in_nodes.size()); i++) {
      const auto& parenti = in_nodes[i]->value;
      value._double *= parenti._double;
    }
  } else {
    throw std::runtime_error(
        "invalid type " + parent0.type.to_string() +
        " for MULTIPLY operator at node_id " + std::to_string(index));
  }
}

LogSumExp::LogSumExp(const std::vector<graph::Node*>& in_nodes)
    : MultiaryOperator(graph::OperatorType::LOGSUMEXP, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::REAL and
      type0 != graph::AtomicType::NEG_REAL and
      type0 != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "operator LOGSUMEXP requires a real or pos/neg_real parent");
  }
  value = graph::NodeValue(graph::AtomicType::REAL);
}

void LogSumExp::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() > 1);
  const graph::NodeValue& parent0 = in_nodes[0]->value;
  if (parent0.type == graph::AtomicType::REAL or
      parent0.type == graph::AtomicType::NEG_REAL or
      parent0.type == graph::AtomicType::POS_REAL) {
    double max_val = parent0._double;
    for (uint i = 1; i < static_cast<uint>(in_nodes.size()); i++) {
      const auto& parenti = in_nodes[i]->value;
      if (parenti._double > max_val) {
        max_val = parenti._double;
      }
    }
    double expsum = 0.0;
    for (const auto parent : in_nodes) {
      expsum += std::exp(parent->value._double - max_val);
    }
    value._double = std::log(expsum) + max_val;
  } else {
    throw std::runtime_error(
        "invalid type " + parent0.type.to_string() +
        " for LOGSUMEXP operator at node_id " + std::to_string(index));
  }
}

Pow::Pow(const std::vector<graph::Node*>& in_nodes)
    : Operator(graph::OperatorType::POW) {
  if (in_nodes.size() != 2) {
    throw std::invalid_argument("operator POW requires 2 parents");
  }
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::PROBABILITY and
      type0 != graph::AtomicType::POS_REAL and
      type0 != graph::AtomicType::REAL) {
    throw std::invalid_argument(
        "operator POW requires a probability, pos_real or real base");
  }
  graph::ValueType type1 = in_nodes[1]->value.type;
  if (type1 != graph::AtomicType::POS_REAL and
      type1 != graph::AtomicType::REAL) {
    throw std::invalid_argument(
        "operator POW requires a pos_real or real exponent");
  }

  // These are all the legal operand types and the result type:
  //
  // R  **  R  -->  R
  // R  **  R+ -->  R
  // R+ **  R  -->  R+
  // R+ **  R+ -->  R+
  // P  **  R  -->  R+  <-- only case where result != type0
  // P  **  R+ -->  P

  graph::AtomicType result = (type0 == graph::AtomicType::PROBABILITY and
                              type1 == graph::AtomicType::REAL)
      ? graph::AtomicType::POS_REAL
      : type0.atomic_type;
  value = graph::NodeValue(result);
}

void Pow::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 2);
  const graph::NodeValue& parent0 = in_nodes[0]->value;
  const graph::NodeValue& parent1 = in_nodes[1]->value;

  if ((parent0.type != graph::AtomicType::REAL and
       parent0.type != graph::AtomicType::POS_REAL and
       parent0.type != graph::AtomicType::PROBABILITY) or
      (parent1.type != graph::AtomicType::REAL and
       parent1.type != graph::AtomicType::POS_REAL)) {
    throw std::runtime_error(
        "invalid type for POW operator at node_id " + std::to_string(index));
  }
  value._double = std::pow(parent0._double, parent1._double);
}

ToMatrix::ToMatrix(const std::vector<graph::Node*>& in_nodes)
    : Operator(graph::OperatorType::TO_MATRIX) {
  if (in_nodes.size() < 3) {
    throw std::invalid_argument(
        "operator TO_MATRIX requires number of rows (m), number of columns (n), "
        "and m * n additional nodes");
  }
  // rows and cols must be constant natural numbers > 0
  if (in_nodes[0]->value.type != graph::AtomicType::NATURAL or
      in_nodes[1]->value.type != graph::AtomicType::NATURAL) {
    throw std::invalid_argument(
        "operator TO_MATRIX requires the first and second parents to be NATURAL"
        "representing the number of rows and the number of columns respectively");
  } else if (
      in_nodes[0]->node_type != graph::NodeType::CONSTANT or
      in_nodes[1]->node_type != graph::NodeType::CONSTANT) {
    throw std::invalid_argument(
        "operator TO_MATRIX requires the number of rows and columns to be CONSTANT");
  } else if (
      (in_nodes[0]->value._natural == 0) or
      (in_nodes[1]->value._natural == 0)) {
    throw std::invalid_argument(
        "operator TO_MATRIX requires the number of rows and columns to be greater than 0");
  }

  uint rows = static_cast<uint>(in_nodes[0]->value._natural);
  uint cols = static_cast<uint>(in_nodes[1]->value._natural);

  if (rows * cols != in_nodes.size() - 2) {
    throw std::invalid_argument(
        "operator TO_MATRIX expected " + std::to_string(rows * cols) +
        "elements in the matrix but received " +
        std::to_string(in_nodes.size() - 2));
  }

  graph::ValueType type0 = in_nodes[2]->value.type;
  for (uint i = 3; i < static_cast<uint>(in_nodes.size()); i++) {
    graph::ValueType type = in_nodes[i]->value.type;
    if (type.variable_type != graph::VariableType::SCALAR) {
      throw std::invalid_argument(
          "operator TO_MATRIX requires scalar nodes as parents");
    } else if (type != type0) {
      throw std::invalid_argument(
          "operator TO_MATRIX requires parent nodes to have the same type");
    }
  }

  value = graph::NodeValue(graph::ValueType(
      graph::VariableType::BROADCAST_MATRIX, type0.atomic_type, rows, cols));
}

void ToMatrix::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() >= 3);
  int rows = static_cast<int>(in_nodes[0]->value._natural);
  int cols = static_cast<int>(in_nodes[1]->value._natural);

  const graph::ValueType& parent_type = in_nodes[2]->value.type;

  if (parent_type == graph::AtomicType::BOOLEAN) {
    Eigen::MatrixXb result(rows, cols);
    for (int j = 0; j < cols; j++) {
      for (int i = 0; i < rows; i++) {
        result(i, j) = in_nodes[2 + j * rows + i]->value._bool;
      }
    }
    value._bmatrix = result;
  } else if (parent_type == graph::AtomicType::NATURAL) {
    Eigen::MatrixXn result(rows, cols);
    for (int j = 0; j < cols; j++) {
      for (int i = 0; i < rows; i++) {
        result(i, j) = in_nodes[2 + j * rows + i]->value._natural;
      }
    }
    value._nmatrix = result;
  } else { // real
    Eigen::MatrixXd result(rows, cols);
    for (int j = 0; j < cols; j++) {
      for (int i = 0; i < rows; i++) {
        result(i, j) = in_nodes[2 + j * rows + i]->value._double;
      }
    }
    value._matrix = result;
  }
}

} // namespace oper
} // namespace beanmachine
