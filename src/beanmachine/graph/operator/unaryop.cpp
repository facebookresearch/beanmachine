/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/unaryop.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace oper {

Complement::Complement(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::COMPLEMENT, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::PROBABILITY and
      type0 != graph::AtomicType::BOOLEAN) {
    throw std::invalid_argument(
        "operator COMPLEMENT requires a boolean or probability parent");
  }
  value = graph::NodeValue(type0);
}

void Complement::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::NodeValue& parent = in_nodes[0]->value;
  if (parent.type == graph::AtomicType::BOOLEAN) {
    value._bool = !parent._bool;
  } else if (parent.type == graph::AtomicType::PROBABILITY) {
    value._double = 1 - parent._double;
  } else {
    throw std::runtime_error(
        "invalid parent type " + parent.type.to_string() +
        " for COMPLEMENT operator at node_id " + std::to_string(index));
  }
}

ToInt::ToInt(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::TO_INT, in_nodes) {
  value = graph::NodeValue(graph::AtomicType::NATURAL);
}

void ToInt::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::NodeValue& parent = in_nodes[0]->value;
  if (parent.type == graph::AtomicType::BOOLEAN) {
    value._double = parent._bool ? 1 : 0;
  } else if (
      parent.type == graph::AtomicType::REAL or
      parent.type == graph::AtomicType::POS_REAL or
      parent.type == graph::AtomicType::NEG_REAL or
      parent.type == graph::AtomicType::PROBABILITY) {
    value._natural = (int)round(parent._double);
  } else if (parent.type == graph::AtomicType::NATURAL) {
    value._natural = parent._natural;
  } else {
    throw std::runtime_error(
        "invalid parent type " + parent.type.to_string() +
        " for TO_INT operator at node_id " + std::to_string(index));
  }
}

ToReal::ToReal(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::TO_REAL, in_nodes) {
  value = graph::NodeValue(graph::AtomicType::REAL);
}

void ToReal::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::NodeValue& parent = in_nodes[0]->value;
  if (parent.type == graph::AtomicType::BOOLEAN) {
    value._double = parent._bool ? 1.0 : 0.0;
  } else if (
      parent.type == graph::AtomicType::REAL or
      parent.type == graph::AtomicType::POS_REAL or
      parent.type == graph::AtomicType::NEG_REAL or
      parent.type == graph::AtomicType::PROBABILITY) {
    value._double = parent._double;
  } else if (parent.type == graph::AtomicType::NATURAL) {
    value._double = (double)parent._natural;
  } else {
    throw std::runtime_error(
        "invalid parent type " + parent.type.to_string() +
        " for TO_REAL operator at node_id " + std::to_string(index));
  }
}

ToRealMatrix::ToRealMatrix(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::TO_REAL_MATRIX, in_nodes) {
  assert(in_nodes.size() == 1);
  const graph::ValueType parent_type = in_nodes[0]->value.type;
  if (parent_type.variable_type != graph::VariableType::BROADCAST_MATRIX and
      parent_type.variable_type != graph::VariableType::COL_SIMPLEX_MATRIX) {
    throw std::invalid_argument(
        "operator TO_REAL_MATRIX requires a matrix parent");
  }
  // There is no further restriction on the input aside from it being
  // a matrix.
  value = graph::NodeValue(graph::ValueType(
      graph::VariableType::BROADCAST_MATRIX,
      graph::AtomicType::REAL,
      parent_type.rows,
      parent_type.cols));
}

void ToRealMatrix::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::Node* parent = in_nodes[0];
  const graph::NodeValue parent_value = parent->value;
  const graph::ValueType parent_type = parent_value.type;

  if (parent_type.variable_type != graph::VariableType::BROADCAST_MATRIX and
      parent_type.variable_type != graph::VariableType::COL_SIMPLEX_MATRIX) {
    throw std::runtime_error(
        "invalid parent type " + parent_type.to_string() +
        " for TO_REAL_MATRIX operator at node_id " + std::to_string(index));
  }

  const graph::AtomicType element_type = parent_type.atomic_type;
  const int rows = static_cast<int>(parent_type.rows);
  const int cols = static_cast<int>(parent_type.cols);

  if (element_type == graph::AtomicType::BOOLEAN) {
    Eigen::MatrixXd result(rows, cols);
    for (int j = 0; j < cols; j++) {
      for (int i = 0; i < rows; i++) {
        result(i, j) = parent_value._bmatrix(i, j) ? 1.0 : 0.0;
      }
    }
    value._matrix = result;
  } else if (element_type == graph::AtomicType::NATURAL) {
    Eigen::MatrixXd result(rows, cols);
    for (int j = 0; j < cols; j++) {
      for (int i = 0; i < rows; i++) {
        result(i, j) = (double)parent_value._nmatrix(i, j);
      }
    }
    value._matrix = result;
  } else {
    assert(
        element_type == graph::AtomicType::REAL or
        element_type == graph::AtomicType::POS_REAL or
        element_type == graph::AtomicType::NEG_REAL or
        element_type == graph::AtomicType::PROBABILITY);
    value._matrix = parent_value._matrix;
  }
}

ToPosReal::ToPosReal(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::TO_POS_REAL, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::PROBABILITY and
      type0 != graph::AtomicType::REAL and
      type0 != graph::AtomicType::POS_REAL and
      type0 != graph::AtomicType::NATURAL and
      type0 != graph::AtomicType::BOOLEAN) {
    throw std::invalid_argument(
        "operator TO_POS_REAL requires a "
        "pos_real, probability, real, natural or boolean parent");
  }
  value = graph::NodeValue(graph::AtomicType::POS_REAL);
}

void ToPosReal::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::NodeValue& parent = in_nodes[0]->value;
  if (parent.type == graph::AtomicType::BOOLEAN) {
    value._double = parent._bool ? 1.0 : 0.0;
  } else if (
      parent.type == graph::AtomicType::POS_REAL or
      parent.type == graph::AtomicType::PROBABILITY) {
    value._double = parent._double;
  } else if (parent.type == graph::AtomicType::REAL) {
    if (parent._double < 0) {
      throw std::runtime_error(
          "invalid value of " + std::to_string(parent._double) +
          " for TO_POS_REAL operator at node_id " + std::to_string(index));
    }
    value._double = parent._double;
  } else if (parent.type == graph::AtomicType::NATURAL) {
    value._double = (double)parent._natural;
  } else {
    throw std::runtime_error(
        "invalid parent type " + parent.type.to_string() +
        " for TO_POS_REAL operator at node_id " + std::to_string(index));
  }
}

ToPosRealMatrix::ToPosRealMatrix(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::TO_POS_REAL_MATRIX, in_nodes) {
  assert(in_nodes.size() == 1);
  const graph::ValueType parent_type = in_nodes[0]->value.type;
  if (parent_type.variable_type != graph::VariableType::BROADCAST_MATRIX and
      parent_type.variable_type != graph::VariableType::COL_SIMPLEX_MATRIX) {
    throw std::invalid_argument(
        "operator TO_POS_REAL_MATRIX requires a matrix parent");
  }

  const graph::AtomicType element_type = parent_type.atomic_type;

  if (element_type != graph::AtomicType::PROBABILITY and
      element_type != graph::AtomicType::POS_REAL and
      element_type != graph::AtomicType::NATURAL and
      element_type != graph::AtomicType::BOOLEAN) {
    throw std::invalid_argument(
        "operator TO_POS_REAL_MATRIX requires a "
        "pos_real, probability, natural or boolean matrix parent");
  }

  value = graph::NodeValue(graph::ValueType(
      graph::VariableType::BROADCAST_MATRIX,
      graph::AtomicType::POS_REAL,
      parent_type.rows,
      parent_type.cols));
}

void ToPosRealMatrix::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::Node* parent = in_nodes[0];
  const graph::NodeValue parent_value = parent->value;
  const graph::ValueType parent_type = parent_value.type;

  if (parent_type.variable_type != graph::VariableType::BROADCAST_MATRIX and
      parent_type.variable_type != graph::VariableType::COL_SIMPLEX_MATRIX) {
    throw std::runtime_error(
        "invalid parent type " + parent_type.to_string() +
        " for TO_POS_REAL_MATRIX operator at node_id " + std::to_string(index));
  }

  const graph::AtomicType element_type = parent_type.atomic_type;
  const int rows = static_cast<int>(parent_type.rows);
  const int cols = static_cast<int>(parent_type.cols);

  if (element_type == graph::AtomicType::BOOLEAN) {
    Eigen::MatrixXd result(rows, cols);
    for (int j = 0; j < cols; j++) {
      for (int i = 0; i < rows; i++) {
        result(i, j) = parent_value._bmatrix(i, j) ? 1.0 : 0.0;
      }
    }
    value._matrix = result;
  } else if (element_type == graph::AtomicType::NATURAL) {
    Eigen::MatrixXd result(rows, cols);
    for (int j = 0; j < cols; j++) {
      for (int i = 0; i < rows; i++) {
        result(i, j) = (double)parent_value._nmatrix(i, j);
      }
    }
    value._matrix = result;
  } else {
    assert(
        element_type == graph::AtomicType::POS_REAL or
        element_type == graph::AtomicType::NEG_REAL or
        element_type == graph::AtomicType::PROBABILITY);
    value._matrix = parent_value._matrix;
  }
}

ToProbability::ToProbability(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::TO_PROBABILITY, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::PROBABILITY and
      type0 != graph::AtomicType::POS_REAL and
      type0 != graph::AtomicType::REAL) {
    throw std::invalid_argument(
        "operator TO_PROBABILITY requires a "
        "real, pos_real, or probability parent");
  }
  value = graph::NodeValue(graph::AtomicType::PROBABILITY);
}

void ToProbability::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::NodeValue& parent = in_nodes[0]->value;
  if (parent.type == graph::AtomicType::PROBABILITY or
      parent.type == graph::AtomicType::POS_REAL or
      parent.type == graph::AtomicType::REAL) {
    // note: we have to cast it to an NodeValue object rather than directly
    // assigning to ensure that the usual boundary checks for probabilities
    // are made
    value = graph::NodeValue(graph::AtomicType::PROBABILITY, parent._double);
  } else {
    throw std::runtime_error(
        "invalid parent type " + parent.type.to_string() +
        " for TO_PROBABILITY operator at node_id " + std::to_string(index));
  }
}

ToNegReal::ToNegReal(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::TO_NEG_REAL, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::NEG_REAL and
      type0 != graph::AtomicType::REAL) {
    throw std::invalid_argument(
        "operator TO_NEG_REAL requires a real or neg_real parent");
  }
  value = graph::NodeValue(graph::AtomicType::NEG_REAL);
}

void ToNegReal::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::NodeValue& parent = in_nodes[0]->value;
  if (parent.type != graph::AtomicType::NEG_REAL and
      parent.type != graph::AtomicType::REAL) {
    throw std::runtime_error(
        "invalid parent type " + parent.type.to_string() +
        " for TO_NEG_REAL operator at node_id " + std::to_string(index));
  }
  // note: we have to cast it to an NodeValue object rather than directly
  // assigning to ensure that the usual boundary checks for negative reals
  // are made
  value = graph::NodeValue(graph::AtomicType::NEG_REAL, parent._double);
}

Negate::Negate(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::NEGATE, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  graph::ValueType new_type;
  if (type0 == graph::AtomicType::REAL) {
    new_type = type0;
  } else if (type0 == graph::AtomicType::POS_REAL) {
    new_type = graph::AtomicType::NEG_REAL;
  } else if (type0 == graph::AtomicType::NEG_REAL) {
    new_type = graph::AtomicType::POS_REAL;
  } else {
    throw std::invalid_argument(
        "operator NEGATE requires a real, pos_real or neg_real parent");
  }
  value = graph::NodeValue(new_type);
}

void Negate::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::NodeValue& parent = in_nodes[0]->value;
  if (parent.type == graph::AtomicType::REAL or
      parent.type == graph::AtomicType::POS_REAL or
      parent.type == graph::AtomicType::NEG_REAL) {
    value._double = -parent._double;
  } else {
    throw std::runtime_error(
        "invalid parent type " + parent.type.to_string() +
        " for NEGATE operator at node_id " + std::to_string(index));
  }
}

Exp::Exp(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::EXP, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  graph::ValueType new_type;
  if (type0 == graph::AtomicType::REAL or
      type0 == graph::AtomicType::POS_REAL) {
    new_type = graph::AtomicType::POS_REAL;
  } else if (type0 == graph::AtomicType::NEG_REAL) {
    new_type = graph::AtomicType::PROBABILITY;
  } else {
    throw std::invalid_argument(
        "operator EXP requires a neg_real, real or pos_real parent");
  }
  value = graph::NodeValue(new_type);
}

void Exp::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::NodeValue& parent = in_nodes[0]->value;
  if (parent.type == graph::AtomicType::REAL or
      parent.type == graph::AtomicType::POS_REAL or
      parent.type == graph::AtomicType::NEG_REAL) {
    value._double = std::exp(parent._double);
  } else {
    throw std::runtime_error(
        "invalid parent type " + parent.type.to_string() +
        " for EXP operator at node_id " + std::to_string(index));
  }
}

ExpM1::ExpM1(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::EXPM1, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::REAL and
      type0 != graph::AtomicType::POS_REAL and
      type0 != graph::AtomicType::NEG_REAL) {
    throw std::invalid_argument(
        "operator EXPM1 requires a real, neg_real or pos_real parent");
  }
  // If the input type is real, positive real or negative real, then the
  // output type of exp(x) - 1 is the same.
  value = graph::NodeValue(type0);
}

void ExpM1::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::NodeValue& parent = in_nodes[0]->value;
  if (parent.type == graph::AtomicType::REAL or
      parent.type == graph::AtomicType::POS_REAL or
      parent.type == graph::AtomicType::NEG_REAL) {
    value._double = std::expm1(parent._double);
  } else {
    throw std::runtime_error(
        "invalid parent type " + parent.type.to_string() +
        " for EXPM1 operator at node_id " + std::to_string(index));
  }
}

Phi::Phi(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::PHI, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::REAL) {
    throw std::invalid_argument("operator PHI requires a real parent");
  }
  value = graph::NodeValue(graph::AtomicType::PROBABILITY);
}

void Phi::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::NodeValue& parent = in_nodes[0]->value;
  assert(parent.type == graph::AtomicType::REAL);
  // note: we have to cast it to an NodeValue object rather than directly
  // assigning to ensure that the usual boundary checks for probabilities
  // are made
  value = graph::NodeValue(
      graph::AtomicType::PROBABILITY, util::Phi(parent._double));
}

Logistic::Logistic(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::LOGISTIC, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::REAL) {
    throw std::invalid_argument("operator LOGISTIC requires a real parent");
  }
  value = graph::NodeValue(graph::AtomicType::PROBABILITY);
}

void Logistic::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::NodeValue& parent = in_nodes[0]->value;
  assert(parent.type == graph::AtomicType::REAL);
  // note: we have to cast it to an NodeValue object rather than directly
  // assigning to ensure that the usual boundary checks for probabilities
  // are made
  value = graph::NodeValue(
      graph::AtomicType::PROBABILITY, util::logistic(parent._double));
}

Log1pExp::Log1pExp(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::LOG1PEXP, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::REAL and
      type0 != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "operator LOG1PEXP requires a real or pos_real parent");
  }
  value = graph::NodeValue(graph::AtomicType::POS_REAL);
}

void Log1pExp::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::NodeValue& parent = in_nodes[0]->value;
  if (parent.type == graph::AtomicType::REAL or
      parent.type == graph::AtomicType::POS_REAL) {
    value._double = util::log1pexp(parent._double);
  } else {
    throw std::runtime_error(
        "invalid parent type " + parent.type.to_string() +
        " for LOG1PEXP operator at node_id " + std::to_string(index));
  }
}

Log::Log(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::LOG, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 == graph::AtomicType::POS_REAL) {
    value = graph::NodeValue(graph::AtomicType::REAL);
  } else if (type0 == graph::AtomicType::PROBABILITY) {
    value = graph::NodeValue(graph::AtomicType::NEG_REAL);
  } else {
    throw std::invalid_argument(
        "operator LOG requires a pos_real or probability parent");
  }
}

void Log::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::NodeValue& parent = in_nodes[0]->value;
  if (parent.type == graph::AtomicType::POS_REAL or
      parent.type == graph::AtomicType::PROBABILITY) {
    value._double = std::log(parent._double);
  } else {
    throw std::runtime_error(
        "invalid parent type " +
        std::to_string(static_cast<int>(parent.type.atomic_type)) +
        " for LOG operator at node_id " + std::to_string(index));
  }
}

Log1mExp::Log1mExp(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::LOG1MEXP, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::NEG_REAL) {
    throw std::invalid_argument("operator LOG1MEXP requires a neg_real parent");
  }
  value = graph::NodeValue(graph::AtomicType::NEG_REAL);
}

void Log1mExp::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::NodeValue& parent = in_nodes[0]->value;
  if (parent.type == graph::AtomicType::NEG_REAL) {
    value._double = util::log1mexp(parent._double);
  } else {
    throw std::runtime_error(
        "invalid parent type " + parent.type.to_string() +
        " for LOG1MEXP operator at node_id " + std::to_string(index));
  }
}

LogSumExpVector::LogSumExpVector(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::LOGSUMEXP_VECTOR, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0.variable_type != graph::VariableType::BROADCAST_MATRIX) {
    throw std::invalid_argument(
        "operator LOGSUMEXP_VECTOR requires a BROADCAST_MATRIX parent");
  }
  if (type0.cols != 1 or type0.rows == 0) {
    throw std::invalid_argument(
        "operator LOGSUMEXP_VECTOR requires a BROADCAST_MATRIX parent"
        "with exactly one column and at least one row");
  }
  if (type0.atomic_type != graph::AtomicType::REAL and
      type0.atomic_type != graph::AtomicType::NEG_REAL and
      type0.atomic_type != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "operator LOGSUMEXP_VECTOR requires a real or pos/neg_real parent");
  }
  value = graph::NodeValue(graph::AtomicType::REAL);
}

void LogSumExpVector::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::NodeValue& parent = in_nodes[0]->value;
  if (parent.type.atomic_type == graph::AtomicType::REAL or
      parent.type.atomic_type == graph::AtomicType::NEG_REAL or
      parent.type.atomic_type == graph::AtomicType::POS_REAL) {
    double max_val = parent._matrix(0);
    for (uint i = 1; i < parent._matrix.size(); i++) {
      double valuei = parent._matrix(i);
      if (valuei > max_val) {
        max_val = valuei;
      }
    }
    double expsum = (parent._matrix.array() - max_val).exp().sum();
    value._double = std::log(expsum) + max_val;
  } else {
    throw std::runtime_error(
        "invalid type " + parent.type.to_string() +
        " for LOGSUMEXP_VECTOR operator at node_id " + std::to_string(index));
  }
}

} // namespace oper
} // namespace beanmachine
