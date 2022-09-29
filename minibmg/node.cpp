/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/node.h"
#include <boost/functional/hash.hpp>
#include <fmt/format.h>
#include <atomic>
#include <cstddef>
#include <stdexcept>

namespace {

using namespace beanmachine::minibmg;

inline std::size_t hash_combine(std::size_t a, std::size_t b) {
  std::size_t seed = 0;
  boost::hash_combine(seed, a);
  boost::hash_combine(seed, b);
  return seed;
}

inline std::size_t hash_combine(std::size_t a, std::size_t b, std::size_t c) {
  std::size_t seed = 0;
  boost::hash_combine(seed, a);
  boost::hash_combine(seed, b);
  boost::hash_combine(seed, c);
  return seed;
}

inline std::size_t hash(const std::vector<Nodep>& in_nodes) {
  return boost::hash_range(in_nodes.begin(), in_nodes.end());
}

} // namespace

namespace beanmachine::minibmg {

std::string make_fresh_rvid() {
  static std::atomic<long> next_rvid = 1;
  return fmt::format("S{}", next_rvid.fetch_add(1));
}

Node::Node(
    const enum Operator op,
    const Type type,
    std::size_t cached_hash_value)
    : op{op}, type{type}, cached_hash_value{cached_hash_value} {}

Node::~Node() {}

OperatorNode::OperatorNode(
    const std::vector<Nodep>& in_nodes,
    const enum Operator op,
    const Type type)
    : Node{op, type, hash_combine((size_t)op, hash(in_nodes))},
      in_nodes{in_nodes} {
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
    : Node{Operator::CONSTANT, Type::REAL, hash_combine((size_t)Operator::CONSTANT, std::hash<double>{}(value))},
      value{value} {}

VariableNode::VariableNode(const std::string& name, const unsigned identifier)
    : Node{Operator::VARIABLE, Type::REAL, hash_combine((size_t)Operator::VARIABLE, std::hash<std::string>{}(name), std::hash<unsigned>{}(identifier))},
      name{name},
      identifier{identifier} {}

SampleNode::SampleNode(Nodep distribution, std::string rvid)
    : Node{Operator::SAMPLE, Type::REAL, hash_combine((size_t)Operator::SAMPLE, distribution->cached_hash_value, std::hash<std::string>{}(rvid))},
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
