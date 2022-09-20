/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/factory.h"

namespace beanmachine::minibmg {

uint Graph::Factory::add_constant(double value) {
  auto sequence = (uint)nodes.size();
  const auto new_node = new ConstantNode{value, sequence};
  nodes.push_back(new_node);
  return sequence;
}

enum Type op_type(enum Operator op) {
  switch (op) {
    case Operator::CONSTANT:
    case Operator::VARIABLE:
    case Operator::ADD:
    case Operator::SUBTRACT:
    case Operator::NEGATE:
    case Operator::MULTIPLY:
    case Operator::DIVIDE:
    case Operator::POW:
    case Operator::EXP:
    case Operator::LOG:
    case Operator::ATAN:
    case Operator::LGAMMA:
    case Operator::POLYGAMMA:
    case Operator::IF_EQUAL:
    case Operator::IF_LESS:
    case Operator::SAMPLE:
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
  std::vector<enum Type> empty{};
  result.reserve((int)Operator::LAST_OPERATOR);
  for (Operator op = (Operator)0; op < Operator::LAST_OPERATOR;
       op = (Operator)((int)op + 1)) {
    result.push_back(empty);
  }
  assert(result.size() == (int)Operator::LAST_OPERATOR);
  result[(uint)Operator::CONSTANT] = {};
  result[(uint)Operator::VARIABLE] = {};
  result[(uint)Operator::ADD] = {Type::REAL, Type::REAL};
  result[(uint)Operator::SUBTRACT] = {Type::REAL, Type::REAL};
  result[(uint)Operator::NEGATE] = {Type::REAL};
  result[(uint)Operator::MULTIPLY] = {Type::REAL, Type::REAL};
  result[(uint)Operator::DIVIDE] = {Type::REAL, Type::REAL};
  result[(uint)Operator::POW] = {Type::REAL, Type::REAL};
  result[(uint)Operator::EXP] = {Type::REAL};
  result[(uint)Operator::LOG] = {Type::REAL};
  result[(uint)Operator::ATAN] = {Type::REAL};
  result[(uint)Operator::LGAMMA] = {Type::REAL};
  result[(uint)Operator::POLYGAMMA] = {Type::REAL, Type::REAL};
  result[(uint)Operator::IF_EQUAL] = {
      Type::REAL, Type::REAL, Type::REAL, Type::REAL};
  result[(uint)Operator::IF_LESS] = {
      Type::REAL, Type::REAL, Type::REAL, Type::REAL};
  result[(uint)Operator::DISTRIBUTION_NORMAL] = {Type::REAL, Type::REAL};
  result[(uint)Operator::DISTRIBUTION_BETA] = {Type::REAL, Type::REAL};
  result[(uint)Operator::DISTRIBUTION_BERNOULLI] = {Type::REAL};
  result[(uint)Operator::SAMPLE] = {Type::DISTRIBUTION};
  result[(uint)Operator::OBSERVE] = {Type::DISTRIBUTION, Type::REAL};
  result[(uint)Operator::QUERY] = {Type::DISTRIBUTION};
  return result;
}

enum Type expected_result_type(enum Operator op) {
  switch (op) {
    case Operator::CONSTANT:
    case Operator::SAMPLE:
    case Operator::ADD:
    case Operator::SUBTRACT:
    case Operator::MULTIPLY:
    case Operator::DIVIDE:
    case Operator::POW:
    case Operator::EXP:
    case Operator::LOG:
    case Operator::ATAN:
    case Operator::LGAMMA:
    case Operator::POLYGAMMA:
    case Operator::IF_EQUAL:
    case Operator::IF_LESS:
      return Type::REAL;

    case Operator::DISTRIBUTION_NORMAL:
    case Operator::DISTRIBUTION_BETA:
    case Operator::DISTRIBUTION_BERNOULLI:
      return Type::DISTRIBUTION;

    case Operator::OBSERVE:
    case Operator::QUERY:
      return Type::NONE;

    default:
      throw std::invalid_argument("Unknown type for operator.");
  }
}

const std::vector<std::vector<enum Type>> expected_parents =
    make_expected_parents();

unsigned arity(Operator op) {
  return expected_parents[(uint)op].size();
}

uint Graph::Factory::add_operator(enum Operator op, std::vector<uint> parents) {
  auto sequence = (uint)nodes.size();
  auto expected = expected_parents[(uint)op];
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
      new OperatorNode{in_nodes, sequence, op, expected_result_type(op)};
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
  auto new_node = new QueryNode{query_id, parent_node, sequence};
  nodes.push_back(new_node);
  return query_id;
}

uint Graph::Factory::add_variable(
    const std::string& name,
    const uint variable_index) {
  auto sequence = (uint)nodes.size();
  auto new_node = new VariableNode{name, variable_index, sequence};
  nodes.push_back(new_node);
  return sequence;
}

Graph Graph::Factory::build() {
  Graph graph{this->nodes};
  this->nodes.clear();
  return graph;
}

Graph::Factory::~Factory() {
  for (auto node : nodes) {
    delete node;
  }
  nodes.clear();
}

} // namespace beanmachine::minibmg
