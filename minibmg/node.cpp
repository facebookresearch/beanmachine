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
  std::size_t seed = 0;
  for (auto n : in_nodes) {
    boost::hash_combine(seed, n->cached_hash_value);
  }
  return seed;
}

} // namespace

namespace beanmachine::minibmg {

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

std::size_t NodepIdentityHash::operator()(
    beanmachine::minibmg::Nodep const& p) const noexcept {
  return p->cached_hash_value;
}

bool NodepIdentityEquals::operator()(
    const beanmachine::minibmg::Nodep& lhs,
    const beanmachine::minibmg::Nodep& rhs) const noexcept {
  const Node* l = lhs.get();
  const Node* r = rhs.get();
  // a node is equal to itself.
  if (l == r) {
    return true;
  }
  // equal nodes have equal hash codes and equal operators.
  if (l == nullptr || r == nullptr ||
      l->cached_hash_value != r->cached_hash_value || l->op != r->op) {
    return false;
  }
  switch (l->op) {
    case Operator::VARIABLE: {
      const VariableNode* vl = dynamic_cast<const VariableNode*>(l);
      const VariableNode* vr = dynamic_cast<const VariableNode*>(r);
      return vl->name == vr->name && vl->identifier == vr->identifier;
    }
    case Operator::CONSTANT: {
      double cl = dynamic_cast<const ConstantNode*>(l)->value;
      double cr = dynamic_cast<const ConstantNode*>(r)->value;
      return std::isnan(cl) ? std::isnan(cr) : cl == cr;
    }
    case Operator::SAMPLE: {
      const SampleNode* sl = dynamic_cast<const SampleNode*>(l);
      const SampleNode* sr = dynamic_cast<const SampleNode*>(r);
      return sl->rvid == sr->rvid &&
          this->operator()(sl->distribution, sr->distribution);
    }
    default: {
      const OperatorNode* lo = dynamic_cast<const OperatorNode*>(l);
      const OperatorNode* ro = dynamic_cast<const OperatorNode*>(r);
      if (lo->in_nodes.size() != ro->in_nodes.size()) {
        return false;
      }
      auto it1 = lo->in_nodes.begin();
      auto it2 = ro->in_nodes.begin();
      for (; it1 != lo->in_nodes.end() && it2 != ro->in_nodes.end();
           it1++, it2++) {
        if (!this->operator()(*it1, *it2)) {
          return false;
        }
      }
      return true;
    }
  }
}

} // namespace beanmachine::minibmg
