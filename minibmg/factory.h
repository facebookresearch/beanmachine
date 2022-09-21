/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>
#include <vector>
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/node.h"
#include "beanmachine/minibmg/operator.h"
#include "beanmachine/minibmg/type.h"

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

class Graph::Factory {
 public:
  NodeId add_constant(double value);

  NodeId add_operator(enum Operator op, std::vector<NodeId> parents);

  // returns the index of the query in the samples, not a NodeId
  unsigned add_query(NodeId parent);
  unsigned add_query(NodeId parent, NodeId& new_node_id);

  NodeId add_variable(const std::string& name, const unsigned variable_index);

  inline Nodep operator[](const NodeId& node_id) const {
    auto t = nodes.find(node_id);
    if (t == nodes.end())
      return nullptr;
    return t->second;
  }
  Graph build();
  ~Factory();

 private:
  std::unordered_map<NodeId, Nodep> nodes;
  std::vector<Nodep> all_nodes;
  unsigned next_query = 0;

  NodeId add_node(Nodep node);
};

enum Type expected_result_type(enum Operator op);
extern const std::vector<std::vector<enum Type>> expected_parents;
unsigned arity(Operator op);
enum Type op_type(enum Operator op);

} // namespace beanmachine::minibmg
