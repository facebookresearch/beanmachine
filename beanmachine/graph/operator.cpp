// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include "beanmachine/graph/binaryop.h"
#include "beanmachine/graph/distribution.h"
#include "beanmachine/graph/operator.h"
#include "beanmachine/graph/unaryop.h"

namespace beanmachine {
namespace oper {

static void check_unary_op(
    graph::OperatorType op_type,
    const std::vector<graph::Node*>& in_nodes) {
  if (in_nodes.size() != 1) {
    throw std::invalid_argument(
      "expecting exactly a single parent for unary operator "
      + std::to_string(static_cast<int>(op_type)));
  }
  // if the parent node's value type has not been initialized then we
  // can't define an operator here
  if (in_nodes[0]->value.type == graph::AtomicType::UNKNOWN) {
    throw std::invalid_argument(
      "unexpected parent node of type "
      + std::to_string(static_cast<int>(in_nodes[0]->node_type))
      + " for operator type "
      + std::to_string(static_cast<int>(op_type))
    );
  }
}

// a multiary op has 2 or more operands
static void check_multiary_op(
    graph::OperatorType op_type,
    const std::vector<graph::Node*>& in_nodes) {
  if (in_nodes.size() < 2) {
    throw std::invalid_argument(
      "expecting at least two parents for operator "
      + std::to_string(static_cast<int>(op_type)));
  }
  // all parent nodes should have a defined value type
  for (const graph::Node* node : in_nodes) {
    if (node->value.type == graph::AtomicType::UNKNOWN) {
      throw std::invalid_argument(
        "unexpected parent node of type "
        + std::to_string(static_cast<int>(node->node_type))
        + " for operator type "
        + std::to_string(static_cast<int>(op_type))
      );
    }
  }
}

Operator::Operator(
    graph::OperatorType op_type,
    const std::vector<graph::Node*>& in_nodes)
    : graph::Node(graph::NodeType::OPERATOR), op_type(op_type) {
  // first check that there is at least one parent and all parents have
  // the same type
  if (in_nodes.size() < 1) {
    throw std::invalid_argument("operator requires a parent");
  }
  graph::AtomicType type0 = in_nodes[0]->value.type;
  for (const graph::Node* node : in_nodes) {
    if (node->value.type != type0) {
      throw std::invalid_argument("all parents of operator should have same type");
    }
  }
  // now perform operator-specific checks and set the value type
  switch (op_type) {
    case graph::OperatorType::SAMPLE: {
      if (in_nodes.size() != 1 or
          in_nodes[0]->node_type != graph::NodeType::DISTRIBUTION) {
        throw std::invalid_argument(
            "~ operator requires a single distribution parent");
      }
      const distribution::Distribution* dist =
        static_cast<distribution::Distribution*>(in_nodes[0]);
      // the type of value of a SAMPLE node is obviously the sample type
      // of the distribution parent
      value.type = dist->sample_type;
      break;
    }
    case graph::OperatorType::COMPLEMENT: {
      check_unary_op(op_type, in_nodes);
      if (type0 != graph::AtomicType::PROBABILITY
          and type0 != graph::AtomicType::BOOLEAN) {
        throw std::invalid_argument(
          "operator COMPLEMENT only supports boolean/probability parent"
        );
      }
      value.type = type0;
      break;
    }
    case graph::OperatorType::TO_REAL: {
      check_unary_op(op_type, in_nodes);
      if (type0 == graph::AtomicType::TENSOR) {
        throw std::invalid_argument(
            "operator TO_REAL doesn't support tensor parent");
      }
      value.type = graph::AtomicType::REAL;
      break;
    }
    case graph::OperatorType::TO_POS_REAL: {
      check_unary_op(op_type, in_nodes);
      if (type0 == graph::AtomicType::REAL
          or type0 == graph::AtomicType::TENSOR) {
        throw std::invalid_argument(
            "operator TO_POS_REAL doesn't support real or tensor parent");
      }
      value.type = graph::AtomicType::POS_REAL;
      break;
    }
    case graph::OperatorType::TO_TENSOR: {
      check_unary_op(op_type, in_nodes);
      value.type = graph::AtomicType::TENSOR;
      break;
    }
    case graph::OperatorType::NEGATE: {
      check_unary_op(op_type, in_nodes);
      if (type0 != graph::AtomicType::REAL
          and type0 != graph::AtomicType::TENSOR) {
        throw std::invalid_argument(
            "operator NEGATE only supports real/tensor parent");
      }
      value.type = type0;
      break;
    }
    case graph::OperatorType::EXPM1:
    case graph::OperatorType::EXP: {
      check_unary_op(op_type, in_nodes);
      if (
          type0 != graph::AtomicType::REAL
          and type0 != graph::AtomicType::POS_REAL
          and type0 != graph::AtomicType::TENSOR) {
        throw std::invalid_argument("operator requires real/tensor parent");
      }
      value.type = type0;
      break;
    }
    case graph::OperatorType::MULTIPLY: {
      check_multiary_op(op_type, in_nodes);
      if (type0 != graph::AtomicType::REAL
          and type0 != graph::AtomicType::POS_REAL
          and type0 != graph::AtomicType::TENSOR
          and type0 != graph::AtomicType::PROBABILITY) {
        throw std::invalid_argument("operator MUTIPLY requires real/tensor/probability parent");
      }
      value.type = type0;
      break;
    }
    case graph::OperatorType::ADD: {
      check_multiary_op(op_type, in_nodes);
      if (type0 != graph::AtomicType::REAL
          and type0 != graph::AtomicType::POS_REAL
          and type0 != graph::AtomicType::TENSOR) {
        throw std::invalid_argument("operator ADD requires real/tensor parent");
      }
      value.type = type0;
      break;
    }
    default: {
      throw std::invalid_argument(
        "Unknown operator " + std::to_string(static_cast<int>(op_type)));
    }
  }
}

void Operator::eval(std::mt19937& gen) {
  if (op_type == graph::OperatorType::SAMPLE) {
    distribution::Distribution* dist =
        static_cast<distribution::Distribution*>(in_nodes[0]);
    value = dist->sample(gen);
    return;
  }

  switch (op_type) {
    case graph::OperatorType::COMPLEMENT: {
      complement(this);
      break;
    }
    case graph::OperatorType::TO_REAL: {
      to_real(this);
      break;
    }
    case graph::OperatorType::TO_POS_REAL: {
      to_pos_real(this);
      break;
    }
    case graph::OperatorType::TO_TENSOR: {
      to_tensor(this);
      break;
    }
    case graph::OperatorType::NEGATE: {
      negate(this);
      break;
    }
    case graph::OperatorType::EXP: {
      exp(this);
      break;
    }
    case graph::OperatorType::EXPM1: {
      expm1(this);
      break;
    }
    case graph::OperatorType::MULTIPLY: {
      multiply(this);
      break;
    }
    case graph::OperatorType::ADD: {
      add(this);
      break;
    }
    default: {
      throw std::runtime_error(
        "internal error: unexpected operator type "
        + std::to_string(static_cast<int>(op_type))
        + " at node_id " + std::to_string(index));
    }
  }
}

} // namespace oper
} // namespace beanmachine
