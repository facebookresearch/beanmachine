/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#pragma once

namespace beanmachine::minibmg {

class Node2;
class ScalarNode2;
class DistributionNode2;
class Node2Visitor;

using Node2p = std::shared_ptr<const Node2>;
using ScalarNode2p = std::shared_ptr<const ScalarNode2>;
using DistributionNode2p = std::shared_ptr<const DistributionNode2>;

class Node2 {
 public:
  std::size_t cached_hash_value;
  virtual void accept(Node2Visitor& visitor) const = 0;
  virtual ~Node2();

 protected:
  explicit Node2(std::size_t cached_hash_value);
};

class ScalarNode2 : public Node2 {
 protected:
  explicit ScalarNode2(std::size_t cached_hash_value);
};

class DistributionNode2 : public Node2 {
 protected:
  explicit DistributionNode2(std::size_t cached_hash_value);
};

std::string make_fresh_rvid();

class ScalarConstantNode2 : public ScalarNode2 {
 public:
  explicit ScalarConstantNode2(double constant_value);
  double constant_value;
  void accept(Node2Visitor& visitor) const override;
};

class ScalarVariableNode2 : public ScalarNode2 {
 public:
  ScalarVariableNode2(const std::string& name, const unsigned identifier);
  std::string name;
  unsigned identifier;
  void accept(Node2Visitor& visitor) const override;
};

class ScalarSampleNode2 : public ScalarNode2 {
 public:
  explicit ScalarSampleNode2(
      const DistributionNode2p& distribution,
      const std::string& rvid = make_fresh_rvid());
  const DistributionNode2p& distribution;
  std::string rvid;
  void accept(Node2Visitor& visitor) const override;
};

class ScalarAddNode2 : public ScalarNode2 {
 public:
  ScalarAddNode2(const ScalarNode2p& left, const ScalarNode2p& right);
  ScalarNode2p left;
  ScalarNode2p right;
  void accept(Node2Visitor& visitor) const override;
};

class ScalarSubtractNode2 : public ScalarNode2 {
 public:
  ScalarSubtractNode2(const ScalarNode2p& left, const ScalarNode2p& right);
  ScalarNode2p left;
  ScalarNode2p right;
  void accept(Node2Visitor& visitor) const override;
};

class ScalarNegateNode2 : public ScalarNode2 {
 public:
  explicit ScalarNegateNode2(const ScalarNode2p& x);
  ScalarNode2p x;
  void accept(Node2Visitor& visitor) const override;
};

class ScalarMultiplyNode2 : public ScalarNode2 {
 public:
  ScalarMultiplyNode2(const ScalarNode2p& left, const ScalarNode2p& right);
  ScalarNode2p left;
  ScalarNode2p right;
  void accept(Node2Visitor& visitor) const override;
};

class ScalarDivideNode2 : public ScalarNode2 {
 public:
  ScalarDivideNode2(const ScalarNode2p& left, const ScalarNode2p& right);
  ScalarNode2p left;
  ScalarNode2p right;
  void accept(Node2Visitor& visitor) const override;
};

class ScalarPowNode2 : public ScalarNode2 {
 public:
  ScalarPowNode2(const ScalarNode2p& left, const ScalarNode2p& right);
  ScalarNode2p left;
  ScalarNode2p right;
  void accept(Node2Visitor& visitor) const override;
};

class ScalarExpNode2 : public ScalarNode2 {
 public:
  explicit ScalarExpNode2(const ScalarNode2p& x);
  ScalarNode2p x;
  void accept(Node2Visitor& visitor) const override;
};

class ScalarLogNode2 : public ScalarNode2 {
 public:
  explicit ScalarLogNode2(const ScalarNode2p& x);
  ScalarNode2p x;
  void accept(Node2Visitor& visitor) const override;
};

class ScalarAtanNode2 : public ScalarNode2 {
 public:
  explicit ScalarAtanNode2(const ScalarNode2p& x);
  ScalarNode2p x;
  void accept(Node2Visitor& visitor) const override;
};

class ScalarLgammaNode2 : public ScalarNode2 {
 public:
  explicit ScalarLgammaNode2(const ScalarNode2p& x);
  ScalarNode2p x;
  void accept(Node2Visitor& visitor) const override;
};

class ScalarPolygammaNode2 : public ScalarNode2 {
 public:
  ScalarPolygammaNode2(const ScalarNode2p& n, const ScalarNode2p& x);
  ScalarNode2p n, x;
  void accept(Node2Visitor& visitor) const override;
};

class ScalarIfEqualNode2 : public ScalarNode2 {
 public:
  ScalarIfEqualNode2(
      const ScalarNode2p& a,
      const ScalarNode2p& b,
      const ScalarNode2p& c,
      const ScalarNode2p& d);
  ScalarNode2p a, b, c, d;
  void accept(Node2Visitor& visitor) const override;
};

class ScalarIfLessNode2 : public ScalarNode2 {
 public:
  ScalarIfLessNode2(
      const ScalarNode2p& a,
      const ScalarNode2p& b,
      const ScalarNode2p& c,
      const ScalarNode2p& d);
  ScalarNode2p a, b, c, d;
  void accept(Node2Visitor& visitor) const override;
};

class DistributionNormalNode2 : public DistributionNode2 {
 public:
  DistributionNormalNode2(const ScalarNode2p& mean, const ScalarNode2p& stddev);
  ScalarNode2p mean;
  ScalarNode2p stddev;
  void accept(Node2Visitor& visitor) const override;
};

class DistributionHalfNormalNode2 : public DistributionNode2 {
 public:
  explicit DistributionHalfNormalNode2(const ScalarNode2p& stddev);
  ScalarNode2p stddev;
  void accept(Node2Visitor& visitor) const override;
};

class DistributionBetaNode2 : public DistributionNode2 {
 public:
  DistributionBetaNode2(const ScalarNode2p& a, const ScalarNode2p& b);
  ScalarNode2p a;
  ScalarNode2p b;
  void accept(Node2Visitor& visitor) const override;
};

class DistributionBernoulliNode2 : public DistributionNode2 {
 public:
  explicit DistributionBernoulliNode2(const ScalarNode2p& prob);
  ScalarNode2p prob;
  void accept(Node2Visitor& visitor) const override;
};

// A helper function useful when topologically sorting ScalarNode2ps (the
// topological_sort function requires a parameter that is a function of this
// shape).
std::vector<Node2p> in_nodes(const Node2p& n);

// Provide a good hash function so ScalarNode2p values can be used in unordered
// maps and sets.  This treats ScalarNode2p values as semantically value-based.
struct Node2pIdentityHash {
  std::size_t operator()(const beanmachine::minibmg::Node2p& p) const noexcept;
};

// Provide a good equality function so ScalarNode2p values can be used in
// unordered maps and sets.  This treats ScalarNode2p values, recursively, as
// semantically value-based.
struct Node2pIdentityEquals {
  bool operator()(
      const beanmachine::minibmg::Node2p& lhs,
      const beanmachine::minibmg::Node2p& rhs) const noexcept;
};

// A value-based map from Node2s to T.  Used for deduplicating and
// optimizing a graph.
class Node2Node2ValueMap {
 public:
  ~Node2Node2ValueMap() {}
  ScalarNode2p at(const ScalarNode2p& p) const {
    return std::dynamic_pointer_cast<const ScalarNode2>(map.at(p));
  }
  DistributionNode2p at(const DistributionNode2p& p) const {
    return std::dynamic_pointer_cast<const DistributionNode2>(map.at(p));
  }
  Node2p at(const Node2p& p) const {
    return map.at(p);
  }
  bool contains(const Node2p& p) const {
    return map.contains(p);
  }
  void add(const Node2p& key, const Node2p& value) {
    map.insert({key, value});
  }

 private:
  std::unordered_map<Node2p, Node2p, Node2pIdentityHash, Node2pIdentityEquals>
      map;
};

template <class T>
using Node2ValueMap =
    std::unordered_map<Node2p, T, Node2pIdentityHash, Node2pIdentityEquals>;

// A value-based set of Node2s.
using Node2ValueSet =
    std::unordered_set<Node2p, Node2pIdentityHash, Node2pIdentityEquals>;

// A visitor for nodes
class Node2Visitor {
 public:
  virtual void visit(const ScalarConstantNode2* node) = 0;
  virtual void visit(const ScalarVariableNode2* node) = 0;
  virtual void visit(const ScalarSampleNode2* node) = 0;
  virtual void visit(const ScalarAddNode2* node) = 0;
  virtual void visit(const ScalarSubtractNode2* node) = 0;
  virtual void visit(const ScalarNegateNode2* node) = 0;
  virtual void visit(const ScalarMultiplyNode2* node) = 0;
  virtual void visit(const ScalarDivideNode2* node) = 0;
  virtual void visit(const ScalarPowNode2* node) = 0;
  virtual void visit(const ScalarExpNode2* node) = 0;
  virtual void visit(const ScalarLogNode2* node) = 0;
  virtual void visit(const ScalarAtanNode2* node) = 0;
  virtual void visit(const ScalarLgammaNode2* node) = 0;
  virtual void visit(const ScalarPolygammaNode2* node) = 0;
  virtual void visit(const ScalarIfEqualNode2* node) = 0;
  virtual void visit(const ScalarIfLessNode2* node) = 0;
  virtual void visit(const DistributionNormalNode2* node) = 0;
  virtual void visit(const DistributionHalfNormalNode2* node) = 0;
  virtual void visit(const DistributionBetaNode2* node) = 0;
  virtual void visit(const DistributionBernoulliNode2* node) = 0;
  virtual ~Node2Visitor() {}
};

// A default visitor for nodes.  Calls the default_visit method for every node
// type except those that are overridden.
class DefaultNode2Visitor : public Node2Visitor {
 public:
  virtual void default_visit(const Node2* node) = 0;

  void visit(const ScalarConstantNode2* node) override;
  void visit(const ScalarVariableNode2* node) override;
  void visit(const ScalarSampleNode2* node) override;
  void visit(const ScalarAddNode2* node) override;
  void visit(const ScalarSubtractNode2* node) override;
  void visit(const ScalarNegateNode2* node) override;
  void visit(const ScalarMultiplyNode2* node) override;
  void visit(const ScalarDivideNode2* node) override;
  void visit(const ScalarPowNode2* node) override;
  void visit(const ScalarExpNode2* node) override;
  void visit(const ScalarLogNode2* node) override;
  void visit(const ScalarAtanNode2* node) override;
  void visit(const ScalarLgammaNode2* node) override;
  void visit(const ScalarPolygammaNode2* node) override;
  void visit(const ScalarIfEqualNode2* node) override;
  void visit(const ScalarIfLessNode2* node) override;
  void visit(const DistributionNormalNode2* node) override;
  void visit(const DistributionHalfNormalNode2* node) override;
  void visit(const DistributionBetaNode2* node) override;
  void visit(const DistributionBernoulliNode2* node) override;
};

// A visitor for scalar nodes.  Throws an exception for distributions.
class ScalarNode2Visitor : public Node2Visitor {
  void default_visit(const Node2* node);
  void visit(const DistributionNormalNode2* node) override;
  void visit(const DistributionHalfNormalNode2* node) override;
  void visit(const DistributionBetaNode2* node) override;
  void visit(const DistributionBernoulliNode2* node) override;
};

// A visitor for distribution nodes.  Throws an exception for scalars.
class DistributionNode2Visitor : public Node2Visitor {
  void default_visit(const Node2* node);
  void visit(const ScalarConstantNode2* node) override;
  void visit(const ScalarVariableNode2* node) override;
  void visit(const ScalarSampleNode2* node) override;
  void visit(const ScalarAddNode2* node) override;
  void visit(const ScalarSubtractNode2* node) override;
  void visit(const ScalarNegateNode2* node) override;
  void visit(const ScalarMultiplyNode2* node) override;
  void visit(const ScalarDivideNode2* node) override;
  void visit(const ScalarPowNode2* node) override;
  void visit(const ScalarExpNode2* node) override;
  void visit(const ScalarLogNode2* node) override;
  void visit(const ScalarAtanNode2* node) override;
  void visit(const ScalarLgammaNode2* node) override;
  void visit(const ScalarPolygammaNode2* node) override;
  void visit(const ScalarIfEqualNode2* node) override;
  void visit(const ScalarIfLessNode2* node) override;
};

} // namespace beanmachine::minibmg
