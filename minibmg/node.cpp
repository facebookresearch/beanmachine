/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/node.h"
#include <stdexcept>

namespace beanmachine::minibmg {

Node::Node(const NodeId sequence, const enum Operator op, const Type type)
    : sequence{sequence}, op{op}, type{type} {}

Node::~Node() {}

OperatorNode::OperatorNode(
    const std::vector<const Node*>& in_nodes,
    const NodeId sequence,
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
    const unsigned query_index,
    const Node* in_node,
    const NodeId sequence)
    : Node{sequence, Operator::QUERY, Type::NONE},
      query_index{query_index},
      in_node{in_node} {}

ConstantNode::ConstantNode(const double value, const NodeId sequence)
    : Node{sequence, Operator::CONSTANT, Type::REAL}, value{value} {}

VariableNode::VariableNode(
    const std::string& name,
    const unsigned variable_index,
    const NodeId sequence)
    : Node{sequence, Operator::VARIABLE, Type::REAL},
      name{name},
      variable_index{variable_index} {}

} // namespace beanmachine::minibmg
