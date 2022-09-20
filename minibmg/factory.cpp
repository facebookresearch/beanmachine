/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/factory.h"
#include <stdexcept>

namespace beanmachine::minibmg {

std::atomic<unsigned long> NodeId::_next_value{0};

NodeId::NodeId() {
  this->value = _next_value.fetch_add(1);
}

NodeId Graph::Factory::add_node(Nodep node) {
  if (built) {
    throw std::invalid_argument("Graph has already been built");
  }
  NodeId sequence{};
  nodes.insert({sequence, node});
  return sequence;
}

NodeId Graph::Factory::add_constant(double value) {
  const auto new_node = std::make_shared<ConstantNode>(value);
  return add_node(new_node);
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
  result[(unsigned)Operator::CONSTANT] = {};
  result[(unsigned)Operator::VARIABLE] = {};
  result[(unsigned)Operator::ADD] = {Type::REAL, Type::REAL};
  result[(unsigned)Operator::SUBTRACT] = {Type::REAL, Type::REAL};
  result[(unsigned)Operator::NEGATE] = {Type::REAL};
  result[(unsigned)Operator::MULTIPLY] = {Type::REAL, Type::REAL};
  result[(unsigned)Operator::DIVIDE] = {Type::REAL, Type::REAL};
  result[(unsigned)Operator::POW] = {Type::REAL, Type::REAL};
  result[(unsigned)Operator::EXP] = {Type::REAL};
  result[(unsigned)Operator::LOG] = {Type::REAL};
  result[(unsigned)Operator::ATAN] = {Type::REAL};
  result[(unsigned)Operator::LGAMMA] = {Type::REAL};
  result[(unsigned)Operator::POLYGAMMA] = {Type::REAL, Type::REAL};
  result[(unsigned)Operator::IF_EQUAL] = {
      Type::REAL, Type::REAL, Type::REAL, Type::REAL};
  result[(unsigned)Operator::IF_LESS] = {
      Type::REAL, Type::REAL, Type::REAL, Type::REAL};
  result[(unsigned)Operator::DISTRIBUTION_NORMAL] = {Type::REAL, Type::REAL};
  result[(unsigned)Operator::DISTRIBUTION_BETA] = {Type::REAL, Type::REAL};
  result[(unsigned)Operator::DISTRIBUTION_BERNOULLI] = {Type::REAL};
  result[(unsigned)Operator::SAMPLE] = {Type::DISTRIBUTION};
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
    case Operator::VARIABLE:
      return Type::REAL;

    case Operator::DISTRIBUTION_NORMAL:
    case Operator::DISTRIBUTION_BETA:
    case Operator::DISTRIBUTION_BERNOULLI:
      return Type::DISTRIBUTION;

    default:
      throw std::invalid_argument("Unknown type for operator.");
  }
}

const std::vector<std::vector<enum Type>> expected_parents =
    make_expected_parents();

unsigned arity(Operator op) {
  return expected_parents[(unsigned)op].size();
}

NodeId Graph::Factory::add_operator(
    enum Operator op,
    std::vector<NodeId> parents) {
  auto expected = expected_parents[(unsigned)op];
  std::vector<Nodep> in_nodes;
  if (parents.size() != expected.size()) {
    throw std::invalid_argument("Incorrect number of parent nodes.");
  }
  for (int i = 0, n = expected.size(); i < n; i++) {
    NodeId p = parents[i];
    auto parent_node = nodes[p];
    if (parent_node == nullptr) {
      throw std::invalid_argument("Reference to nonexistent node.");
    }
    if (parent_node->type != expected[i]) {
      throw std::invalid_argument("Incorrect type for parent node.");
    }
    in_nodes.push_back(parent_node);
  }

  auto new_node =
      std::make_shared<OperatorNode>(in_nodes, op, expected_result_type(op));
  return add_node(new_node);
}

NodeId Graph::Factory::add_variable(
    const std::string& name,
    const unsigned variable_index) {
  auto new_node = std::make_shared<VariableNode>(name, variable_index);
  return add_node(new_node);
}

void Graph::Factory::add_observation(NodeId sample, double value) {
  auto parent_node = nodes[sample];
  if (parent_node == nullptr) {
    throw std::invalid_argument("Reference to nonexistent node.");
  }
  this->observations[parent_node] = value;
}

unsigned Graph::Factory::add_query(NodeId queried) {
  auto parent_node = nodes[queried];
  if (parent_node == nullptr) {
    throw std::invalid_argument("Reference to nonexistent node.");
  }
  if (parent_node->type != Type::REAL) {
    throw std::invalid_argument("Can only query a scalar.");
  }
  auto result = (unsigned)this->queries.size();
  this->queries.push_back(parent_node);
  return result;
}

Graph Graph::Factory::build() {
  if (built) {
    throw std::invalid_argument("Graph has already been built");
  }

  // We preserve this->nodes so it can be used for lookup.
  built = true;
  return Graph::create(queries, observations);
}

Graph::Factory::~Factory() {
  this->nodes.clear();
}

} // namespace beanmachine::minibmg
