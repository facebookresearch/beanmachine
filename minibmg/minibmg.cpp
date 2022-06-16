/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/minibmg.h"
#include <cassert>
#include <stdexcept>

namespace beanmachine::minibmg {

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
  result.reserve(Operator::LAST);
  for (Operator op = (Operator)0; op < Operator::LAST;
       op = (Operator)(op + 1)) {
    result.push_back(empty);
  }
  assert(result.size() == Operator::LAST);
  result[Operator::CONSTANT] = {};
  result[Operator::ADD] = {Type::REAL, Type::REAL};
  result[Operator::MULTIPLY] = {Type::REAL, Type::REAL};
  result[Operator::DISTRIBUTION_NORMAL] = {Type::REAL, Type::REAL};
  result[Operator::DISTRIBUTION_BETA] = {Type::REAL, Type::REAL};
  result[Operator::DISTRIBUTION_BERNOULLI] = {Type::REAL};
  result[Operator::SAMPLE] = {Type::DISTRIBUTION};
  result[Operator::OBSERVE] = {Type::DISTRIBUTION, Type::REAL};
  result[Operator::QUERY] = {Type::REAL};
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
  switch (op) {
    case Operator::CONSTANT:
      throw std::invalid_argument("Use add_constant to add a constant.");
    case Operator::QUERY:
      throw std::invalid_argument("Use add_query to add a query.");
    default:;
  }
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
      query_id, {parent_node}, sequence, Operator::QUERY, Type::NONE);
  nodes.push_back(new_node);
  return query_id;
}

const Node* Graph::Factory::get_node(uint node_index) {
  auto num_nodes = (uint)nodes.size();
  if (node_index >= num_nodes) {
    throw std::invalid_argument("Reference to nonexistent node.");
  }
  return nodes[node_index];
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
  Graph graph = Graph(nodes);
  graph.validate();
  return graph;
}

void Graph::validate() {
  // TODO: not implemented.
}

} // namespace beanmachine::minibmg
