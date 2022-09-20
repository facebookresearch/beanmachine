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

class Node {
 public:
  Node(const uint sequence, const enum Operator op, const Type type);
  const uint sequence;
  const enum Operator op;
  const enum Type type;
  virtual ~Node() = 0;
};

class OperatorNode : public Node {
 public:
  OperatorNode(
      const std::vector<const Node*>& in_nodes,
      const uint sequence,
      const enum Operator op,
      const enum Type type);
  const std::vector<const Node*> in_nodes;
};

class ConstantNode : public Node {
 public:
  ConstantNode(const double value, const uint sequence);
  const double value;
};

class VariableNode : public Node {
 public:
  VariableNode(
      const std::string& name,
      const uint variable_index,
      const uint sequence);
  const std::string name;
  const uint variable_index;
};

class QueryNode : public Node {
 public:
  QueryNode(const uint query_index, const Node* in_node, const uint sequence);
  const uint query_index;
  const Node* const in_node;
};

} // namespace beanmachine::minibmg
