/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/minibmg.h"
#include <folly/Format.h>
#include <cassert>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace beanmachine::minibmg {

std::unordered_map<Operator, std::string> operator_names;
std::unordered_map<std::string, Operator> string_to_operator;

bool _c0 = [] {
  auto add = [](Operator op, std::string name) {
    operator_names[op] = name;
    string_to_operator[name] = op;
  };
  add(Operator::CONSTANT, "CONSTANT");
  add(Operator::ADD, "ADD");
  add(Operator::MULTIPLY, "MULTIPLY");
  add(Operator::DISTRIBUTION_NORMAL, "DISTRIBUTION_NORMAL");
  add(Operator::DISTRIBUTION_BETA, "DISTRIBUTION_BETA");
  add(Operator::DISTRIBUTION_BERNOULLI, "DISTRIBUTION_BERNOULLI");
  add(Operator::SAMPLE, "SAMPLE");
  add(Operator::OBSERVE, "OBSERVE");
  add(Operator::QUERY, "QUERY");
  return true;
}();

Operator operator_from_name(std::string name) {
  auto found = string_to_operator.find(name);
  if (found != string_to_operator.end()) {
    return found->second;
  }

  return Operator::NO_OPERATOR;
}

std::string to_string(Operator op) {
  if (op < 0 || op >= Operator::LAST_OPERATOR) {
    return "NO_OPERATOR";
  }
  auto found = operator_names.find(op);
  if (found != operator_names.end()) {
    return found->second;
  }

  return "NO_OPERATOR";
}

std::unordered_map<Type, std::string> type_names;
std::unordered_map<std::string, Type> string_to_type;

bool _c1 = [] {
  auto add = [](Type type, std::string name) {
    type_names[type] = name;
    string_to_type[name] = type;
  };
  add(Type::REAL, "REAL");
  add(Type::DISTRIBUTION, "DISTRIBUTION");
  add(Type::NONE, "NONE");
  return true;
}();

Type type_from_name(std::string name) {
  auto found = string_to_type.find(name);
  if (found != string_to_type.end()) {
    return found->second;
  }

  return Type::NONE;
}

std::string to_string(Type type) {
  if (type < 0 || type >= Type::LAST_TYPE) {
    return "NONE";
  }
  auto found = type_names.find(type);
  if (found != type_names.end()) {
    return found->second;
  }

  return "NONE";
}

Node::Node(const uint sequence, const enum Operator op, const Type type)
    : sequence(sequence), op(op), type(type) {}

Node::~Node() {}

OperatorNode::OperatorNode(
    const std::vector<const Node*>& in_nodes,
    const uint sequence,
    const enum Operator op,
    const Type type)
    : Node(sequence, op, type), in_nodes(in_nodes) {
  switch (op) {
    case Operator::CONSTANT:
      throw std::invalid_argument("OperatorNode cannot be used for CONSTANT.");
    case Operator::QUERY:
      throw std::invalid_argument("OperatorNode cannot be used for QUERY.");
    default:;
  }
}

QueryNode::QueryNode(
    const uint query_index,
    const Node* in_node,
    const uint sequence,
    const enum Operator op,
    const Type type)
    : Node(sequence, op, type), query_index(query_index), in_node(in_node) {
  switch (op) {
    case Operator::QUERY:
      break;
    case Operator::CONSTANT:
      throw std::invalid_argument("QueryNode cannot be used for CONSTANT.");
    default:
      throw std::invalid_argument("QueryNode cannot be used for an operator.");
  }
}

ConstantNode::ConstantNode(
    const double value,
    const uint sequence,
    const enum Operator op,
    const Type type)
    : Node(sequence, op, type), value(value) {
  switch (op) {
    case Operator::CONSTANT:
      break;
    case Operator::QUERY:
      throw std::invalid_argument("ConstantNode cannot be used for QUERY.");
    default:
      throw std::invalid_argument(
          "ConstantNode cannot be used for an operator.");
  }
}

uint Graph::Factory::add_constant(double value) {
  auto sequence = (uint)nodes.size();
  const auto new_node =
      new ConstantNode(value, sequence, Operator::CONSTANT, Type::REAL);
  nodes.push_back(new_node);
  return sequence;
}

enum Type op_type(enum Operator op) {
  switch (op) {
    case Operator::CONSTANT:
    case Operator::SAMPLE:
    case Operator::ADD:
    case Operator::MULTIPLY:
      return Type::REAL;
    case Operator::DISTRIBUTION_NORMAL:
    case Operator::DISTRIBUTION_BETA:
    case Operator::DISTRIBUTION_BERNOULLI:
      return Type::DISTRIBUTION;
    case Operator::OBSERVE:
    case Operator::QUERY:
      return Type::NONE;
    default:
      throw std::invalid_argument("op_type not defined for operator.");
  }
}

const std::vector<std::vector<enum Type>> make_expected_parents() {
  std::vector<std::vector<enum Type>> result;
  std::vector<enum Type> empty = {};
  result.reserve(Operator::LAST_OPERATOR);
  for (Operator op = (Operator)0; op < Operator::LAST_OPERATOR;
       op = (Operator)(op + 1)) {
    result.push_back(empty);
  }
  assert(result.size() == Operator::LAST_OPERATOR);
  result[Operator::CONSTANT] = {};
  result[Operator::ADD] = {Type::REAL, Type::REAL};
  result[Operator::MULTIPLY] = {Type::REAL, Type::REAL};
  result[Operator::DISTRIBUTION_NORMAL] = {Type::REAL, Type::REAL};
  result[Operator::DISTRIBUTION_BETA] = {Type::REAL, Type::REAL};
  result[Operator::DISTRIBUTION_BERNOULLI] = {Type::REAL};
  result[Operator::SAMPLE] = {Type::DISTRIBUTION};
  result[Operator::OBSERVE] = {Type::DISTRIBUTION, Type::REAL};
  result[Operator::QUERY] = {Type::DISTRIBUTION};
  return result;
}

enum Type expected_result_type(enum Operator op) {
  switch (op) {
    case CONSTANT:
    case SAMPLE:
    case ADD:
    case MULTIPLY:
      return Type::REAL;

    case DISTRIBUTION_NORMAL:
    case DISTRIBUTION_BETA:
    case DISTRIBUTION_BERNOULLI:
      return Type::DISTRIBUTION;

    case OBSERVE:
    case QUERY:
      return Type::NONE;

    default:
      throw std::invalid_argument("Unknown type for operator.");
  }
}

const std::vector<std::vector<enum Type>> expected_parents =
    make_expected_parents();

uint Graph::Factory::add_operator(enum Operator op, std::vector<uint> parents) {
  auto sequence = (uint)nodes.size();
  auto expected = expected_parents[op];
  std::vector<const Node*> in_nodes;
  if (parents.size() != expected.size()) {
    throw std::invalid_argument("Incorrect number of parent nodes.");
  }
  for (int i = 0, n = expected.size(); i < n; i++) {
    uint p = parents[i];
    if (p >= sequence) {
      throw std::invalid_argument("Reference to nonexistent node.");
    }
    auto parent_node = nodes[p];
    if (parent_node->type != expected[i]) {
      throw std::invalid_argument("Incorrect type for parent node.");
    }
    in_nodes.push_back(parent_node);
  }

  auto new_node =
      new OperatorNode(in_nodes, sequence, op, expected_result_type(op));
  nodes.push_back(new_node);
  return sequence;
}

uint Graph::Factory::add_query(uint parent) {
  auto sequence = (uint)nodes.size();
  if (parent >= sequence) {
    throw std::invalid_argument("Reference to nonexistent node.");
  }
  auto parent_node = nodes[parent];
  if (parent_node->type != Type::DISTRIBUTION) {
    throw std::invalid_argument("Incorrect parent for QUERY node.");
  }
  auto query_id = next_query;
  next_query++;
  auto new_node = new QueryNode(
      query_id, parent_node, sequence, Operator::QUERY, Type::NONE);
  nodes.push_back(new_node);
  return query_id;
}

Graph Graph::Factory::build() {
  auto graph = Graph(this->nodes);
  this->nodes.clear();
  return graph;
}

Graph::Factory::~Factory() {
  for (auto node : nodes) {
    delete node;
  }
  nodes.clear();
}

Graph::Graph(std::vector<const Node*> nodes) : nodes(nodes) {}

Graph::~Graph() {
  for (auto node : nodes) {
    delete node;
  }
}

Graph Graph::create(std::vector<const Node*> nodes) {
  Graph::validate(nodes);
  return Graph(nodes);
}

void Graph::validate(std::vector<const Node*> nodes) {
  std::unordered_set<const Node*> seen;
  uint next_query = 0;
  // Check the nodes.
  for (int i = 0, n = nodes.size(); i < n; i++) {
    auto node = nodes[i];

    // Check that the nodes are in sequence.
    if (node->sequence != i) {
      throw std::invalid_argument(fmt::format(
          "Node {0} has sequence number {1} but should be {0}",
          i,
          node->sequence));
    }

    // Check that the operator is in range.
    if (node->op < (Operator)0 || node->op >= Operator::LAST_OPERATOR) {
      throw std::invalid_argument(
          fmt::format("Node {0} has invalid operator {1}", i, (int)node->op));
    }

    // Check the node type
    if (node->type != expected_result_type(node->op)) {
      throw std::invalid_argument(fmt::format(
          "Node {0} has type {1} but should be {2}",
          i,
          to_string(node->type),
          to_string(expected_result_type(node->op))));
    }

    // Check the predecessor nodes
    switch (node->op) {
      case CONSTANT:
        break;
      case QUERY: {
        const QueryNode* q = (QueryNode*)node;
        if (q->query_index != next_query) {
          throw std::invalid_argument(fmt::format(
              "Node {0} has query index {1} but should be {2}",
              i,
              q->query_index,
              next_query));
        }
        next_query++;
        if (!seen.count(q->in_node)) {
          throw std::invalid_argument(
              fmt::format("Query Node {0} parent not previously seen", i));
        }
        if (q->in_node->type == Type::DISTRIBUTION) {
          throw std::invalid_argument(fmt::format(
              "Query Node {0} should have a distribution input", i));
        }
        break;
      }

      // Check other operators.
      default: {
        const OperatorNode* op = (OperatorNode*)node;
        auto parent_types = expected_parents[node->op];
        if (op->in_nodes.size() != parent_types.size()) {
          throw std::invalid_argument(fmt::format(
              "Node {0} should have {1} parents", i, parent_types.size()));
        }
        for (int j = 0, m = parent_types.size(); j < m; j++) {
          const Node* parent = op->in_nodes[j];
          if (!seen.count(parent)) {
            throw std::invalid_argument(
                fmt::format("Node {0} has a parent not previously seen", i));
          }
          if (parent->type != parent_types[j]) {
            throw std::invalid_argument(fmt::format(
                "Node {0} should have a {1} input",
                i,
                to_string(parent_types[j])));
          }
        }
      }
    }

    seen.insert(node);
  }
}

} // namespace beanmachine::minibmg
