/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/node.h"
#include <atomic>
#include <stdexcept>

namespace {

std::atomic<long> next_rvid = 1;
long make_fresh_rvid() {
  return next_rvid.fetch_add(1);
}

} // namespace

namespace beanmachine::minibmg {

Node::Node(const enum Operator op, const Type type) : op{op}, type{type} {}

Node::~Node() {}

OperatorNode::OperatorNode(
    const std::vector<Nodep>& in_nodes,
    const enum Operator op,
    const Type type)
    : Node{op, type}, in_nodes{in_nodes} {
  switch (op) {
    case Operator::CONSTANT:
    case Operator::VARIABLE:
    case Operator::SAMPLE:
      throw std::invalid_argument(
          "OperatorNode cannot be used for " + to_string(op) + ".");
    default:;
  }
}

ConstantNode::ConstantNode(const double value)
    : Node{Operator::CONSTANT, Type::REAL}, value{value} {}

VariableNode::VariableNode(const std::string& name, const unsigned identifier)
    : Node{Operator::VARIABLE, Type::REAL},
      name{name},
      identifier{identifier} {}

SampleNode::SampleNode(Nodep distribution)
    : Node{Operator::SAMPLE, Type::REAL},
      distribution{distribution},
      rvid{make_fresh_rvid()} {}

SampleNode::SampleNode(Nodep distribution, long rvid)
    : Node{Operator::SAMPLE, Type::REAL},
      distribution{distribution},
      rvid{rvid} {}

std::vector<Nodep> in_nodes(const Nodep& n) {
  switch (n->op) {
    case Operator::CONSTANT:
    case Operator::VARIABLE:
      return {};
    case Operator::SAMPLE:
      return {std::dynamic_pointer_cast<const SampleNode>(n)->distribution};
    default:
      return std::dynamic_pointer_cast<const OperatorNode>(n)->in_nodes;
  }
}

} // namespace beanmachine::minibmg
