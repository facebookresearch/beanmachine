/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <vector>
#include "beanmachine/minibmg/operator.h"
#include "beanmachine/minibmg/type.h"

#pragma once

namespace beanmachine::minibmg {

// TODO: replace this with an opaque identifier.
using NodeId = unsigned;

class Node {
 public:
  Node(const NodeId sequence, const enum Operator op, const Type type);
  const NodeId sequence;
  const enum Operator op;
  const enum Type type;
  virtual ~Node() = 0;
};

class OperatorNode : public Node {
 public:
  OperatorNode(
      const std::vector<const Node*>& in_nodes,
      const NodeId sequence,
      const enum Operator op,
      const enum Type type);
  const std::vector<const Node*> in_nodes;
};

class ConstantNode : public Node {
 public:
  ConstantNode(const double value, const NodeId sequence);
  const double value;
};

class VariableNode : public Node {
 public:
  VariableNode(
      const std::string& name,
      const unsigned variable_index,
      const NodeId sequence);
  const std::string name;
  const unsigned variable_index;
};

class QueryNode : public Node {
 public:
  QueryNode(
      const unsigned query_index,
      const Node* in_node,
      const NodeId sequence);
  const unsigned query_index;
  const Node* const in_node;
};

} // namespace beanmachine::minibmg
