/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/node.h"
#include <stdexcept>

namespace beanmachine::minibmg {

Node::Node(const uint sequence, const enum Operator op, const Type type)
    : sequence{sequence}, op{op}, type{type} {}

Node::~Node() {}

OperatorNode::OperatorNode(
    const std::vector<const Node*>& in_nodes,
    const uint sequence,
    const enum Operator op,
    const Type type)
    : Node{sequence, op, type}, in_nodes{in_nodes} {
  switch (op) {
    case Operator::CONSTANT:
    case Operator::QUERY:
    case Operator::VARIABLE:
      throw std::invalid_argument(
          "OperatorNode cannot be used for " + to_string(op) + ".");
    default:;
  }
}

QueryNode::QueryNode(
    const uint query_index,
    const Node* in_node,
    const uint sequence)
    : Node{sequence, Operator::QUERY, Type::NONE},
      query_index{query_index},
      in_node{in_node} {}

ConstantNode::ConstantNode(const double value, const uint sequence)
    : Node{sequence, Operator::CONSTANT, Type::REAL}, value{value} {}

VariableNode::VariableNode(
    const std::string& name,
    const uint variable_index,
    const uint sequence)
    : Node{sequence, Operator::VARIABLE, Type::REAL},
      name{name},
      variable_index{variable_index} {}

} // namespace beanmachine::minibmg
