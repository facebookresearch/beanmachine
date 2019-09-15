// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include "beanmachine/graph/binaryop.h"
#include "beanmachine/graph/distribution.h"
#include "beanmachine/graph/operator.h"
#include "beanmachine/graph/unaryop.h"

namespace beanmachine {
namespace oper {

Operator::Operator(
    graph::OperatorType op_type,
    const std::vector<graph::Node*>& in_nodes)
    : graph::Node(graph::NodeType::OPERATOR), op_type(op_type) {
  // check parent nodes are of the correct type, here the ~ operator has
  // very different requirements than the other operators
  if (op_type == graph::OperatorType::SAMPLE) {
    if (in_nodes.size() != 1 or
        in_nodes[0]->node_type != graph::NodeType::DISTRIBUTION) {
      throw std::invalid_argument(
          "~ operator requires a single distribution parent");
    }
  } else {
    for (graph::Node* parent : in_nodes) {
      if (parent->node_type != graph::NodeType::CONSTANT and
          parent->node_type != graph::NodeType::OPERATOR) {
        throw std::invalid_argument(
            "operator parent must be a constant or another operator");
      }
    }
  }
  switch (op_type) {
    case graph::OperatorType::SAMPLE: {
      break;
    }
    case graph::OperatorType::TO_REAL:
    case graph::OperatorType::NEGATE:
    case graph::OperatorType::EXP: {
      if (in_nodes.size() != 1) {
        throw std::invalid_argument(
          "expecting exactly a single parent for unary operator "
          + std::to_string(static_cast<int>(op_type)));
      }
      break;
    }
    case graph::OperatorType::MULTIPLY:
    case graph::OperatorType::ADD: {
      if (in_nodes.size() < 2) {
        throw std::invalid_argument(
          "expecting at least two parents for operator "
          + std::to_string(static_cast<int>(op_type)));
      }
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
    case graph::OperatorType::TO_REAL: {
      to_real(this);
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
