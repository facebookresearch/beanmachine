// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

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
        "operator COMPLEMENT only supports boolean/probability parent");
  }
  value = graph::AtomicValue(type0);
}

void Complement::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::AtomicValue& parent = in_nodes[0]->value;
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

ToReal::ToReal(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::TO_REAL, in_nodes) {
  value = graph::AtomicValue(graph::AtomicType::REAL);
}

void ToReal::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::AtomicValue& parent = in_nodes[0]->value;
  if (parent.type == graph::AtomicType::BOOLEAN) {
    value._double = parent._bool ? 1.0 : 0.0;
  } else if (
      parent.type == graph::AtomicType::REAL or
      parent.type == graph::AtomicType::POS_REAL or
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

ToPosReal::ToPosReal(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::TO_POS_REAL, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 == graph::AtomicType::REAL) {
    throw std::invalid_argument(
        "operator TO_POS_REAL doesn't support real parent");
  }
  value = graph::AtomicValue(graph::AtomicType::POS_REAL);
}

void ToPosReal::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::AtomicValue& parent = in_nodes[0]->value;
  if (parent.type == graph::AtomicType::BOOLEAN) {
    value._double = parent._bool ? 1.0 : 0.0;
  } else if (
      parent.type == graph::AtomicType::POS_REAL or
      parent.type == graph::AtomicType::PROBABILITY) {
    value._double = parent._double;
  } else if (parent.type == graph::AtomicType::NATURAL) {
    value._double = (double)parent._natural;
  } else {
    throw std::runtime_error(
        "invalid parent type " + parent.type.to_string() +
        " for TO_POS_REAL operator at node_id " + std::to_string(index));
  }
}

Negate::Negate(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::NEGATE, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::REAL) {
    throw std::invalid_argument("operator NEGATE only supports real parent");
  }
  value = graph::AtomicValue(type0);
}

void Negate::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::AtomicValue& parent = in_nodes[0]->value;
  if (parent.type == graph::AtomicType::REAL) {
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
  if (type0 != graph::AtomicType::REAL and
      type0 != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument("operator EXP requires real/pos_real parent");
  }
  value = graph::AtomicValue(graph::AtomicType::POS_REAL);
}

void Exp::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::AtomicValue& parent = in_nodes[0]->value;
  if (parent.type == graph::AtomicType::REAL or
      parent.type == graph::AtomicType::POS_REAL) {
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
      type0 != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument("operator EXPM1 requires real/pos_real parent");
  }
  // pos_real -> e^x - 1 -> pos_real
  // real -> e^x - 1 -> real
  value = graph::AtomicValue(type0);
}

void ExpM1::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::AtomicValue& parent = in_nodes[0]->value;
  if (parent.type == graph::AtomicType::REAL or
      parent.type == graph::AtomicType::POS_REAL) {
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
    throw std::invalid_argument("Phi require a real-valued parent");
  }
  value.type = graph::AtomicType::PROBABILITY;
}

void Phi::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::AtomicValue& parent = in_nodes[0]->value;
  assert(parent.type == graph::AtomicType::REAL);
  // note: we have to cast it to an AtomicValue object rather than directly
  // assigning to ensure that the usual boundary checks for probabilities
  // are made
  value = graph::AtomicValue(
      graph::AtomicType::PROBABILITY, util::Phi(parent._double));
}

Logistic::Logistic(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::LOGISTIC, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::REAL) {
    throw std::invalid_argument("logistic require a real-valued parent");
  }
  value.type = graph::AtomicType::PROBABILITY;
}

void Logistic::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::AtomicValue& parent = in_nodes[0]->value;
  assert(parent.type == graph::AtomicType::REAL);
  // note: we have to cast it to an AtomicValue object rather than directly
  // assigning to ensure that the usual boundary checks for probabilities
  // are made
  value = graph::AtomicValue(
      graph::AtomicType::PROBABILITY, util::logistic(parent._double));
}

Log1pExp::Log1pExp(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::LOG1PEXP, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::REAL and
      type0 != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "operator LOG1PEXP requires real/pos_real parent");
  }
  value = graph::AtomicValue(graph::AtomicType::POS_REAL);
}

void Log1pExp::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::AtomicValue& parent = in_nodes[0]->value;
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
  if (type0 != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument("operator LOG requires a pos_real parent");
  }
  value = graph::AtomicValue(graph::AtomicType::REAL);
}

void Log::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::AtomicValue& parent = in_nodes[0]->value;
  if (parent.type == graph::AtomicType::POS_REAL) {
    value._double = std::log(parent._double);
  } else {
    throw std::runtime_error(
        "invalid parent type " +
        std::to_string(static_cast<int>(parent.type.atomic_type)) +
        " for LOG operator at node_id " + std::to_string(index));
  }
}

NegativeLog::NegativeLog(const std::vector<graph::Node*>& in_nodes)
    : UnaryOperator(graph::OperatorType::NEGATIVE_LOG, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 == graph::AtomicType::POS_REAL) {
    value = graph::AtomicValue(graph::AtomicType::REAL);
  } else if (type0 == graph::AtomicType::PROBABILITY) {
    value = graph::AtomicValue(graph::AtomicType::POS_REAL);
  } else {
    throw std::invalid_argument(
        "operator NEG_LOG requires a pos_real/probability parent");
  }
}

void NegativeLog::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 1);
  const graph::AtomicValue& parent = in_nodes[0]->value;
  if (parent.type == graph::AtomicType::POS_REAL or
      parent.type == graph::AtomicType::PROBABILITY) {
    value._double = -std::log(parent._double);
  } else {
    throw std::runtime_error(
        "invalid parent type " +
        std::to_string(static_cast<int>(parent.type.atomic_type)) +
        " for NEGATIVE_LOG operator at node_id " + std::to_string(index));
  }
}

} // namespace oper
} // namespace beanmachine
