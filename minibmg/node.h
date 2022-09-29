/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <boost/functional/hash.hpp>
#include <fmt/format.h>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>
#include "beanmachine/minibmg/operator.h"
#include "beanmachine/minibmg/type.h"

#pragma once

namespace beanmachine::minibmg {

class Node;

using Nodep = std::shared_ptr<const Node>;

class Node {
 public:
  Node(const enum Operator op, const Type type, std::size_t cached_hash_value);
  enum Operator op;
  enum Type type;
  std::size_t cached_hash_value;
  virtual ~Node() = 0;
};

class OperatorNode : public Node {
 public:
  OperatorNode(
      const std::vector<Nodep>& in_nodes,
      const enum Operator op,
      const enum Type type);
  std::vector<Nodep> in_nodes;
};

class ConstantNode : public Node {
 public:
  explicit ConstantNode(const double value);
  double value;
};

// Variables are identified by name and/or a number identifier.  The mapping
// from variables to values or storage locations is defined by the client of
// this API.
class VariableNode : public Node {
 public:
  VariableNode(const std::string& name, const unsigned identifier);
  std::string name;
  unsigned identifier;
};

std::string make_fresh_rvid();

class SampleNode : public Node {
 public:
  explicit SampleNode(Nodep distribution, std::string rvid = make_fresh_rvid());
  Nodep distribution;
  // We assign a distinct ID to each sample operation in a model
  std::string rvid;
};

std::vector<Nodep> in_nodes(const Nodep& n);

} // namespace beanmachine::minibmg

// Provide a good hash function so Nodep values can be used in unordered maps
// and sets.  This treats Nodep values as semantically value-based.
template <>
struct std::hash<beanmachine::minibmg::Nodep> {
  std::size_t operator()(beanmachine::minibmg::Nodep const& p) const noexcept {
    return p->cached_hash_value;
  }
};

// Provide a good equality function so Nodep values can be used in unordered
// maps and sets.  This treats Nodep values, recursively, as semantically
// value-based.
template <>
struct std::equal_to<beanmachine::minibmg::Nodep> {
  bool operator()(
      const beanmachine::minibmg::Nodep& lhs,
      const beanmachine::minibmg::Nodep& rhs) const {
    using namespace beanmachine::minibmg;
    const Node* l = lhs.get();
    const Node* r = rhs.get();
    // a node is equal to itself.
    if (l == r) {
      return true;
    }
    // equal nodes have equal hash codes and equal operators.
    if (l->cached_hash_value != r->cached_hash_value || l->op != r->op) {
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
};
