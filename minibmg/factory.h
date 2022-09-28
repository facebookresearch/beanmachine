/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <list>
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
  explicit NodeId(unsigned long value) : value{value} {}

  inline bool operator==(const NodeId& other) const {
    return value == other.value;
  }
  NodeId(const NodeId& other) : value{other.value} {} // copy ctor
  ~NodeId() {} // dtor

  inline unsigned long _value() const {
    return value;
  }

 private:
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
  NodeId add_sample(NodeId distribution);

  // Add an observation to the graph.  The sample must identify a SAMPLE node.
  void add_observation(NodeId sample, double value);

  // returns the index of the query in the samples, not a NodeId
  unsigned add_query(NodeId parent);

  NodeId add_variable(const std::string& name, const unsigned variable_index);

  inline Nodep operator[](const NodeId& node_id) const {
    auto t = identifer_to_node.find(node_id);
    if (t == identifer_to_node.end()) {
      return nullptr;
    }
    return t->second;
  }
  Graph build();
  ~Factory();

 private:
  bool built = false;
  std::unordered_map<NodeId, Nodep> identifer_to_node;
  std::vector<Nodep> queries;
  std::list<std::pair<Nodep, double>> observations;
  unsigned long next_identifier = 0;

  NodeId add_node(Nodep node);
};

enum Type expected_result_type(enum Operator op);
extern const std::vector<std::vector<enum Type>> expected_parents;
unsigned arity(Operator op);
enum Type op_type(enum Operator op);

} // namespace beanmachine::minibmg
