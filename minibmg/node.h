/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/format.h>
#include <atomic>
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
  Node(const enum Operator op, const Type type);
  enum Operator op;
  enum Type type;
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

class SampleNode : public Node {
 public:
  explicit SampleNode(Nodep distribution);
  explicit SampleNode(Nodep distribution, std::string rvid);
  Nodep distribution;
  // We assign a distinct ID to each sample operation in a model
  std::string rvid;
};

std::vector<Nodep> in_nodes(const Nodep& n);

} // namespace beanmachine::minibmg
