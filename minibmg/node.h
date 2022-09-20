/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/format.h>
#include <atomic>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>
#include "beanmachine/minibmg/operator.h"
#include "beanmachine/minibmg/type.h"

#pragma once

namespace beanmachine::minibmg {

class Node;

// An opaque identifier for a node.
class NodeId {
 public:
  // Create a fresh, new, never-before-seen NodeId
  NodeId();
  explicit NodeId(unsigned long value) : value{value} {}
  explicit NodeId(long value) : value{(unsigned long)value} {}
  inline bool operator==(const NodeId& other) const {
    return value == other.value;
  }
  NodeId(const NodeId& other) : value{other.value} {} // copy ctor
  NodeId& operator=(const NodeId& other) { // assignment
    this->value = other.value;
    return *this;
  }
  ~NodeId() {} // dtor

  inline unsigned long _value() const {
    return value;
  }

  static void _reset_for_testing() {
    _next_value = 0;
  }

 private:
  static std::atomic<unsigned long> _next_value;
  unsigned long value;
};

} // namespace beanmachine::minibmg

// Make NodeId values usable as a key in a hash table.
template <>
struct std::hash<beanmachine::minibmg::NodeId> {
  std::size_t operator()(const beanmachine::minibmg::NodeId& n) const noexcept {
    return (std::size_t)n._value();
  }
};

// Make NodeId values printable using format.
template <>
struct fmt::formatter<beanmachine::minibmg::NodeId>
    : fmt::formatter<std::string> {
  auto format(const beanmachine::minibmg::NodeId& n, format_context& ctx) {
    return formatter<std::string>::format(fmt::format("{}", n._value()), ctx);
  }
};

namespace beanmachine::minibmg {

class Node {
 public:
  Node(const enum Operator op, const Type type);
  NodeId sequence;
  enum Operator op;
  enum Type type;
  virtual ~Node() = 0;
};

class OperatorNode : public Node {
 public:
  OperatorNode(
      const std::vector<const Node*>& in_nodes,
      const enum Operator op,
      const enum Type type);
  std::vector<const Node*> in_nodes;
};

class ConstantNode : public Node {
 public:
  ConstantNode(const double value);
  double value;
};

class VariableNode : public Node {
 public:
  VariableNode(const std::string& name, const unsigned variable_index);
  std::string name;
  unsigned variable_index;
};

class QueryNode : public Node {
 public:
  QueryNode(const unsigned query_index, const Node* in_node);
  unsigned query_index;
  const Node* in_node;
};

} // namespace beanmachine::minibmg
