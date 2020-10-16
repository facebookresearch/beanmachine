// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include "beanmachine/graph/operator/multiaryop.h"

namespace beanmachine {
namespace oper {

Add::Add(const std::vector<graph::Node*>& in_nodes)
    : MultiaryOperator(graph::OperatorType::ADD, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::REAL and
      type0 != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "operator ADD requires real/pos_real parent");
  }
  value = graph::AtomicValue(type0);
}

void Add::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() > 1);
  const graph::AtomicValue& parent0 = in_nodes[0]->value;
  if (parent0.type == graph::AtomicType::REAL or
      parent0.type == graph::AtomicType::POS_REAL) {
    value._double = parent0._double;

    for (uint i = 1; i < in_nodes.size(); i++) {
      const auto& parenti = in_nodes[i]->value;
      value._double += parenti._double;
    }
  } else {
    throw std::runtime_error(
        "invalid type " + parent0.type.to_string() +
        " for ADD operator at node_id " + std::to_string(index));
  }
}


Multiply::Multiply(const std::vector<graph::Node*>& in_nodes)
    : MultiaryOperator(graph::OperatorType::MULTIPLY, in_nodes) {
  graph::ValueType type0 = in_nodes[0]->value.type;
  if (type0 != graph::AtomicType::REAL and
      type0 != graph::AtomicType::POS_REAL and
      type0 != graph::AtomicType::PROBABILITY) {
    throw std::invalid_argument(
        "operator MUTIPLY requires a real, pos_real or probability parent");
  }
  value = graph::AtomicValue(type0);
}

void Multiply::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() > 1);
  const graph::AtomicValue& parent0 = in_nodes[0]->value;
  if (parent0.type == graph::AtomicType::REAL or
      parent0.type == graph::AtomicType::POS_REAL or
      parent0.type == graph::AtomicType::PROBABILITY) {
    value._double = parent0._double;

    for (uint i = 1; i < in_nodes.size(); i++) {
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
      type0 != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "operator LOGSUMEXP requires a real or pos_real parent");
  }
  value = graph::AtomicValue(graph::AtomicType::REAL);
}

void LogSumExp::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() > 1);
  const graph::AtomicValue& parent0 = in_nodes[0]->value;
  if (parent0.type == graph::AtomicType::REAL or
      parent0.type == graph::AtomicType::POS_REAL) {
    double max_val = parent0._double;
    for (uint i = 1; i < in_nodes.size(); i++) {
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
    : Operator(graph::OperatorType::POW)  {
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
  value = graph::AtomicValue(result);
}

void Pow::eval(std::mt19937& /* gen */) {
  assert(in_nodes.size() == 2);
  const graph::AtomicValue& parent0 = in_nodes[0]->value;
  const graph::AtomicValue& parent1 = in_nodes[1]->value;

  if ((parent0.type != graph::AtomicType::REAL and
       parent0.type != graph::AtomicType::POS_REAL and
       parent0.type != graph::AtomicType::PROBABILITY) or
      (parent1.type != graph::AtomicType::REAL and
       parent1.type != graph::AtomicType::POS_REAL)) {
    throw std::runtime_error(
        "invalid type for POW operator at node_id " +
        std::to_string(index));
  }
  value._double = std::pow(parent0._double, parent1._double);
}

} // namespace oper
} // namespace beanmachine
