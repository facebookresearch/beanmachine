/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace beanmachine::minibmg {

class Node;
class ScalarNode;
class DistributionNode;
class ScalarSampleNode;
class NodeVisitor;

using Nodep = std::shared_ptr<const Node>;
using ScalarNodep = std::shared_ptr<const ScalarNode>;
using DistributionNodep = std::shared_ptr<const DistributionNode>;
using ScalarSampleNodep = std::shared_ptr<const ScalarSampleNode>;

class Node {
 public:
  std::size_t cached_hash_value;
  virtual void accept(NodeVisitor& visitor) const = 0;
  virtual ~Node();

 protected:
  explicit Node(std::size_t cached_hash_value);
};

class ScalarNode : public Node {
 protected:
  explicit ScalarNode(std::size_t cached_hash_value);
};

class DistributionNode : public Node {
 protected:
  explicit DistributionNode(std::size_t cached_hash_value);
};

std::string make_fresh_rvid();

class ScalarConstantNode : public ScalarNode {
 public:
  explicit ScalarConstantNode(double constant_value);
  double constant_value;
  void accept(NodeVisitor& visitor) const override;
};

class ScalarVariableNode : public ScalarNode {
 public:
  ScalarVariableNode(const std::string& name, const unsigned identifier);
  std::string name;
  unsigned identifier;
  void accept(NodeVisitor& visitor) const override;
};

class ScalarSampleNode : public ScalarNode {
 public:
  explicit ScalarSampleNode(
      const DistributionNodep& distribution,
      const std::string& rvid = make_fresh_rvid());
  const DistributionNodep distribution;
  std::string rvid;
  void accept(NodeVisitor& visitor) const override;
};

class ScalarAddNode : public ScalarNode {
 public:
  ScalarAddNode(const ScalarNodep& left, const ScalarNodep& right);
  ScalarNodep left;
  ScalarNodep right;
  void accept(NodeVisitor& visitor) const override;
};

class ScalarSubtractNode : public ScalarNode {
 public:
  ScalarSubtractNode(const ScalarNodep& left, const ScalarNodep& right);
  ScalarNodep left;
  ScalarNodep right;
  void accept(NodeVisitor& visitor) const override;
};

class ScalarNegateNode : public ScalarNode {
 public:
  explicit ScalarNegateNode(const ScalarNodep& x);
  ScalarNodep x;
  void accept(NodeVisitor& visitor) const override;
};

class ScalarMultiplyNode : public ScalarNode {
 public:
  ScalarMultiplyNode(const ScalarNodep& left, const ScalarNodep& right);
  ScalarNodep left;
  ScalarNodep right;
  void accept(NodeVisitor& visitor) const override;
};

class ScalarDivideNode : public ScalarNode {
 public:
  ScalarDivideNode(const ScalarNodep& left, const ScalarNodep& right);
  ScalarNodep left;
  ScalarNodep right;
  void accept(NodeVisitor& visitor) const override;
};

class ScalarPowNode : public ScalarNode {
 public:
  ScalarPowNode(const ScalarNodep& left, const ScalarNodep& right);
  ScalarNodep left;
  ScalarNodep right;
  void accept(NodeVisitor& visitor) const override;
};

class ScalarExpNode : public ScalarNode {
 public:
  explicit ScalarExpNode(const ScalarNodep& x);
  ScalarNodep x;
  void accept(NodeVisitor& visitor) const override;
};

class ScalarLogNode : public ScalarNode {
 public:
  explicit ScalarLogNode(const ScalarNodep& x);
  ScalarNodep x;
  void accept(NodeVisitor& visitor) const override;
};

class ScalarAtanNode : public ScalarNode {
 public:
  explicit ScalarAtanNode(const ScalarNodep& x);
  ScalarNodep x;
  void accept(NodeVisitor& visitor) const override;
};

class ScalarLgammaNode : public ScalarNode {
 public:
  explicit ScalarLgammaNode(const ScalarNodep& x);
  ScalarNodep x;
  void accept(NodeVisitor& visitor) const override;
};

class ScalarPolygammaNode : public ScalarNode {
 public:
  ScalarPolygammaNode(const ScalarNodep& n, const ScalarNodep& x);
  ScalarNodep n, x;
  void accept(NodeVisitor& visitor) const override;
};

class ScalarLog1pNode : public ScalarNode {
 public:
  ScalarLog1pNode(const ScalarNodep& x);
  ScalarNodep n, x;
  void accept(NodeVisitor& visitor) const override;
};

class ScalarIfEqualNode : public ScalarNode {
 public:
  ScalarIfEqualNode(
      const ScalarNodep& a,
      const ScalarNodep& b,
      const ScalarNodep& c,
      const ScalarNodep& d);
  ScalarNodep a, b, c, d;
  void accept(NodeVisitor& visitor) const override;
};

class ScalarIfLessNode : public ScalarNode {
 public:
  ScalarIfLessNode(
      const ScalarNodep& a,
      const ScalarNodep& b,
      const ScalarNodep& c,
      const ScalarNodep& d);
  ScalarNodep a, b, c, d;
  void accept(NodeVisitor& visitor) const override;
};

class DistributionNormalNode : public DistributionNode {
 public:
  DistributionNormalNode(const ScalarNodep& mean, const ScalarNodep& stddev);
  ScalarNodep mean;
  ScalarNodep stddev;
  void accept(NodeVisitor& visitor) const override;
};

class DistributionHalfNormalNode : public DistributionNode {
 public:
  explicit DistributionHalfNormalNode(const ScalarNodep& stddev);
  ScalarNodep stddev;
  void accept(NodeVisitor& visitor) const override;
};

class DistributionBetaNode : public DistributionNode {
 public:
  DistributionBetaNode(const ScalarNodep& a, const ScalarNodep& b);
  ScalarNodep a;
  ScalarNodep b;
  void accept(NodeVisitor& visitor) const override;
};

class DistributionBernoulliNode : public DistributionNode {
 public:
  explicit DistributionBernoulliNode(const ScalarNodep& prob);
  ScalarNodep prob;
  void accept(NodeVisitor& visitor) const override;
};

// A helper function useful when topologically sorting ScalarNodeps (the
// topological_sort function requires a parameter that is a function of this
// shape).
std::vector<Nodep> in_nodes(const Nodep& n);

std::string to_string(const Nodep& node);

// Provide a good hash function so ScalarNodep values can be used in unordered
// maps and sets.  This treats ScalarNodep values as semantically value-based.
struct NodepIdentityHash {
  std::size_t operator()(const beanmachine::minibmg::Nodep& p) const noexcept;
};

// Provide a good equality function so ScalarNodep values can be used in
// unordered maps and sets.  This treats ScalarNodep values, recursively, as
// semantically value-based.
struct NodepIdentityEquals {
  bool operator()(
      const beanmachine::minibmg::Nodep& lhs,
      const beanmachine::minibmg::Nodep& rhs) const noexcept;
};

// A value-based map from Nodes to T.  Used for deduplicating and
// optimizing a graph.
class NodeNodeValueMap {
 private:
  std::unordered_map<Nodep, Nodep, NodepIdentityHash, NodepIdentityEquals> map;

 public:
  ~NodeNodeValueMap() {}
  ScalarNodep at(const ScalarNodep& p) const {
    return std::dynamic_pointer_cast<const ScalarNode>(map.at(p));
  }
  DistributionNodep at(const DistributionNodep& p) const {
    return std::dynamic_pointer_cast<const DistributionNode>(map.at(p));
  }
  Nodep at(const Nodep& p) const {
    return map.at(p);
  }
  bool contains(const Nodep& p) const {
    return map.contains(p);
  }
  void insert(const Nodep& key, const Nodep& value) {
    map.insert({key, value});
  }
  auto find(const Nodep& key) {
    return map.find(key);
  }
  auto end() {
    return map.end();
  }
};

// A visitor for nodes
class NodeVisitor {
 public:
  virtual void visit(const ScalarConstantNode* node) = 0;
  virtual void visit(const ScalarVariableNode* node) = 0;
  virtual void visit(const ScalarSampleNode* node) = 0;
  virtual void visit(const ScalarAddNode* node) = 0;
  virtual void visit(const ScalarSubtractNode* node) = 0;
  virtual void visit(const ScalarNegateNode* node) = 0;
  virtual void visit(const ScalarMultiplyNode* node) = 0;
  virtual void visit(const ScalarDivideNode* node) = 0;
  virtual void visit(const ScalarPowNode* node) = 0;
  virtual void visit(const ScalarExpNode* node) = 0;
  virtual void visit(const ScalarLogNode* node) = 0;
  virtual void visit(const ScalarAtanNode* node) = 0;
  virtual void visit(const ScalarLgammaNode* node) = 0;
  virtual void visit(const ScalarPolygammaNode* node) = 0;
  virtual void visit(const ScalarLog1pNode* node) = 0;
  virtual void visit(const ScalarIfEqualNode* node) = 0;
  virtual void visit(const ScalarIfLessNode* node) = 0;
  virtual void visit(const DistributionNormalNode* node) = 0;
  virtual void visit(const DistributionHalfNormalNode* node) = 0;
  virtual void visit(const DistributionBetaNode* node) = 0;
  virtual void visit(const DistributionBernoulliNode* node) = 0;
  virtual ~NodeVisitor() {}
};

// A default visitor for nodes.  Calls the default_visit method for every node
// type except those that are overridden.
class DefaultNodeVisitor : public NodeVisitor {
 public:
  virtual void default_visit(const Node* node) = 0;

  void visit(const ScalarConstantNode* node) override;
  void visit(const ScalarVariableNode* node) override;
  void visit(const ScalarSampleNode* node) override;
  void visit(const ScalarAddNode* node) override;
  void visit(const ScalarSubtractNode* node) override;
  void visit(const ScalarNegateNode* node) override;
  void visit(const ScalarMultiplyNode* node) override;
  void visit(const ScalarDivideNode* node) override;
  void visit(const ScalarPowNode* node) override;
  void visit(const ScalarExpNode* node) override;
  void visit(const ScalarLogNode* node) override;
  void visit(const ScalarAtanNode* node) override;
  void visit(const ScalarLgammaNode* node) override;
  void visit(const ScalarPolygammaNode* node) override;
  void visit(const ScalarLog1pNode* node) override;
  void visit(const ScalarIfEqualNode* node) override;
  void visit(const ScalarIfLessNode* node) override;
  void visit(const DistributionNormalNode* node) override;
  void visit(const DistributionHalfNormalNode* node) override;
  void visit(const DistributionBetaNode* node) override;
  void visit(const DistributionBernoulliNode* node) override;
};

// A visitor for scalar nodes.  Throws an exception for distributions.
class ScalarNodeVisitor : public NodeVisitor {
  void default_visit(const Node* node);
  void visit(const DistributionNormalNode* node) override;
  void visit(const DistributionHalfNormalNode* node) override;
  void visit(const DistributionBetaNode* node) override;
  void visit(const DistributionBernoulliNode* node) override;
};

// A visitor for distribution nodes.  Throws an exception for scalars.
class DistributionNodeVisitor : public NodeVisitor {
  void default_visit(const Node* node);
  void visit(const ScalarConstantNode* node) override;
  void visit(const ScalarVariableNode* node) override;
  void visit(const ScalarSampleNode* node) override;
  void visit(const ScalarAddNode* node) override;
  void visit(const ScalarSubtractNode* node) override;
  void visit(const ScalarNegateNode* node) override;
  void visit(const ScalarMultiplyNode* node) override;
  void visit(const ScalarDivideNode* node) override;
  void visit(const ScalarPowNode* node) override;
  void visit(const ScalarExpNode* node) override;
  void visit(const ScalarLogNode* node) override;
  void visit(const ScalarAtanNode* node) override;
  void visit(const ScalarLgammaNode* node) override;
  void visit(const ScalarPolygammaNode* node) override;
  void visit(const ScalarLog1pNode* node) override;
  void visit(const ScalarIfEqualNode* node) override;
  void visit(const ScalarIfLessNode* node) override;
};

} // namespace beanmachine::minibmg
