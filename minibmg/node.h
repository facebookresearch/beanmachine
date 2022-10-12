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
#include <unordered_map>
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

// A helper function useful when topologically sorting nodes (the
// topological_sort function requires a parameter that is a function of this
// shape).
std::vector<Nodep> in_nodes(const Nodep& n);

// Provide a good hash function so Nodep values can be used in unordered maps
// and sets.  This treats Nodep values as semantically value-based.
struct NodepIdentityHash {
  std::size_t operator()(beanmachine::minibmg::Nodep const& p) const noexcept;
};

// Provide a good equality function so Nodep values can be used in unordered
// maps and sets.  This treats Nodep values, recursively, as semantically
// value-based.
struct NodepIdentityEquals {
  bool operator()(
      const beanmachine::minibmg::Nodep& lhs,
      const beanmachine::minibmg::Nodep& rhs) const noexcept;
};

// A value-based map from nodes to T.  Used for deduplicating and optimizing a
// graph.
template <class T>
using NodeValueMap =
    std::unordered_map<Nodep, T, NodepIdentityHash, NodepIdentityEquals>;

// A value-based set of nodes.
using NodeValueSet =
    std::unordered_set<Nodep, NodepIdentityHash, NodepIdentityEquals>;

} // namespace beanmachine::minibmg
