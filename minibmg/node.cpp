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
#include <typeinfo>
#include "beanmachine/minibmg/pretty.h"

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

inline std::size_t hash_combine(const std::vector<std::size_t>& many) {
  std::size_t seed = 0;
  for (auto n : many) {
    boost::hash_combine(seed, n);
  }
  return seed;
}

class in_node_gatherer : public NodeVisitor {
 private:
  in_node_gatherer() {}
  static in_node_gatherer instance;

 public:
  static std::vector<Nodep> gather(const Nodep& n) {
    n->accept(instance);
    auto result = instance.result;
    instance.result.clear();
    return result;
  }
  std::vector<Nodep> result;
  void visit(const ScalarConstantNode*) override {
    result = {};
  }
  void visit(const ScalarVariableNode*) override {
    result = {};
  }
  void visit(const ScalarSampleNode* n) override {
    result = {n->distribution};
  }
  void visit(const ScalarAddNode* n) override {
    result = {n->left, n->right};
  }
  void visit(const ScalarSubtractNode* n) override {
    result = {n->left, n->right};
  }
  void visit(const ScalarNegateNode* n) override {
    result = {n->x};
  }
  void visit(const ScalarMultiplyNode* n) override {
    result = {n->left, n->right};
  }
  void visit(const ScalarDivideNode* n) override {
    result = {n->left, n->right};
  }
  void visit(const ScalarPowNode* n) override {
    result = {n->left, n->right};
  }
  void visit(const ScalarExpNode* n) override {
    result = {n->x};
  }
  void visit(const ScalarLogNode* n) override {
    result = {n->x};
  }
  void visit(const ScalarAtanNode* n) override {
    result = {n->x};
  }
  void visit(const ScalarLgammaNode* n) override {
    result = {n->x};
  }
  void visit(const ScalarPolygammaNode* n) override {
    result = {n->n, n->x};
  }
  void visit(const ScalarLog1pNode* n) override {
    result = {n->x};
  }
  void visit(const ScalarIfEqualNode* n) override {
    result = {n->a, n->b, n->c, n->d};
  }
  void visit(const ScalarIfLessNode* n) override {
    result = {n->a, n->b, n->c, n->d};
  }
  void visit(const DistributionNormalNode* n) override {
    result = {n->mean, n->stddev};
  }
  void visit(const DistributionHalfNormalNode* n) override {
    result = {n->stddev};
  }
  void visit(const DistributionBetaNode* n) override {
    result = {n->a, n->b};
  }
  void visit(const DistributionBernoulliNode* n) override {
    result = {n->prob};
  }
  void visit(const DistributionExponentialNode* n) override {
    result = {n->rate};
  }
};

in_node_gatherer in_node_gatherer::instance{};

} // namespace

namespace beanmachine::minibmg {

Node::Node(std::size_t cached_hash_value)
    : cached_hash_value{cached_hash_value} {}

Node::~Node() {}

std::string to_string(const Nodep& node) {
  auto pretty_result = pretty_print({node});
  std::stringstream code;
  for (auto p : pretty_result.prelude) {
    code << p << std::endl;
  }
  code << pretty_result.code[node];
  return code.str();
}

ScalarNode::ScalarNode(std::size_t cached_hash_value)
    : Node{cached_hash_value} {}

DistributionNode::DistributionNode(std::size_t cached_hash_value)
    : Node{cached_hash_value} {}

std::string make_fresh_rvid() {
  static std::atomic<long> next_rvid = 1;
  return fmt::format("S{}", next_rvid.fetch_add(1));
}

ScalarConstantNode::ScalarConstantNode(double constant_value)
    : ScalarNode{hash_combine(
          std::hash<std::string>{}("ScalarConstantNode"),
          std::hash<double>{}(constant_value))},
      constant_value(constant_value) {}

void ScalarConstantNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

ScalarVariableNode::ScalarVariableNode(
    const std::string& name,
    const unsigned identifier)
    : ScalarNode{hash_combine(
          std::hash<std::string>{}("ScalarVariableNode"),
          std::hash<std::string>{}(name),
          std::hash<unsigned>{}(identifier))},
      name{name},
      identifier{identifier} {}

void ScalarVariableNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

ScalarSampleNode::ScalarSampleNode(
    const DistributionNodep& distribution,
    const std::string& rvid)
    : ScalarNode{hash_combine(
          std::hash<std::string>{}("ScalarSampleNode"),
          distribution->cached_hash_value,
          std::hash<std::string>{}(rvid))},
      distribution{distribution},
      rvid{rvid} {}

void ScalarSampleNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

ScalarAddNode::ScalarAddNode(const ScalarNodep& left, const ScalarNodep& right)
    : ScalarNode{hash_combine(
          std::hash<std::string>{}("ScalarAddNode"),
          left->cached_hash_value,
          right->cached_hash_value)},
      left{left},
      right{right} {}

void ScalarAddNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

ScalarSubtractNode::ScalarSubtractNode(
    const ScalarNodep& left,
    const ScalarNodep& right)
    : ScalarNode{hash_combine(
          std::hash<std::string>{}("ScalarSubtractNode"),
          left->cached_hash_value,
          right->cached_hash_value)},
      left{left},
      right{right} {}

void ScalarSubtractNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

ScalarNegateNode::ScalarNegateNode(const ScalarNodep& x)
    : ScalarNode{hash_combine(
          std::hash<std::string>{}("ScalarNegateNode"),
          x->cached_hash_value)},
      x{x} {}

void ScalarNegateNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

ScalarMultiplyNode::ScalarMultiplyNode(
    const ScalarNodep& left,
    const ScalarNodep& right)
    : ScalarNode{hash_combine(
          std::hash<std::string>{}("ScalarMultiplyNode"),
          left->cached_hash_value,
          right->cached_hash_value)},
      left{left},
      right{right} {}

void ScalarMultiplyNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

ScalarDivideNode::ScalarDivideNode(
    const ScalarNodep& left,
    const ScalarNodep& right)
    : ScalarNode{hash_combine(
          std::hash<std::string>{}("ScalarDivideNode"),
          left->cached_hash_value,
          right->cached_hash_value)},
      left{left},
      right{right} {}

void ScalarDivideNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

ScalarPowNode::ScalarPowNode(const ScalarNodep& left, const ScalarNodep& right)
    : ScalarNode{hash_combine(
          std::hash<std::string>{}("ScalarPowNode"),
          left->cached_hash_value,
          right->cached_hash_value)},
      left{left},
      right{right} {}

void ScalarPowNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

ScalarExpNode::ScalarExpNode(const ScalarNodep& x)
    : ScalarNode{hash_combine(
          std::hash<std::string>{}("ScalarExpNode"),
          x->cached_hash_value)},
      x{x} {}

void ScalarExpNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

ScalarLogNode::ScalarLogNode(const ScalarNodep& x)
    : ScalarNode{hash_combine(
          std::hash<std::string>{}("ScalarLogNode"),
          x->cached_hash_value)},
      x{x} {}

void ScalarLogNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

ScalarAtanNode::ScalarAtanNode(const ScalarNodep& x)
    : ScalarNode{hash_combine(
          std::hash<std::string>{}("ScalarAtanNode"),
          x->cached_hash_value)},
      x{x} {}

void ScalarAtanNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

ScalarLgammaNode::ScalarLgammaNode(const ScalarNodep& x)
    : ScalarNode{hash_combine(
          std::hash<std::string>{}("ScalarLgammaNode"),
          x->cached_hash_value)},
      x{x} {}

void ScalarLgammaNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

ScalarPolygammaNode::ScalarPolygammaNode(
    const ScalarNodep& n,
    const ScalarNodep& x)
    : ScalarNode{hash_combine(
          std::hash<std::string>{}("ScalarPolygammaNode"),
          n->cached_hash_value,
          x->cached_hash_value)},
      n{n},
      x{x} {}

void ScalarPolygammaNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

ScalarLog1pNode::ScalarLog1pNode(const ScalarNodep& x)
    : ScalarNode{hash_combine(
          std::hash<std::string>{}("ScalarLog1pNode"),
          x->cached_hash_value)},
      x{x} {}

void ScalarLog1pNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

ScalarIfEqualNode::ScalarIfEqualNode(
    const ScalarNodep& a,
    const ScalarNodep& b,
    const ScalarNodep& c,
    const ScalarNodep& d)
    : ScalarNode{hash_combine(
          {std::hash<std::string>{}("ScalarIfEqualNode"),
           a->cached_hash_value,
           b->cached_hash_value,
           c->cached_hash_value,
           d->cached_hash_value})},
      a{a},
      b{b},
      c{c},
      d{d} {}

void ScalarIfEqualNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

ScalarIfLessNode::ScalarIfLessNode(
    const ScalarNodep& a,
    const ScalarNodep& b,
    const ScalarNodep& c,
    const ScalarNodep& d)
    : ScalarNode{hash_combine(
          {std::hash<std::string>{}("ScalarIfLessNode"),
           a->cached_hash_value,
           b->cached_hash_value,
           c->cached_hash_value,
           d->cached_hash_value})},
      a{a},
      b{b},
      c{c},
      d{d} {}

void ScalarIfLessNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

DistributionNormalNode::DistributionNormalNode(
    const ScalarNodep& mean,
    const ScalarNodep& stddev)
    : DistributionNode{hash_combine(
          std::hash<std::string>{}("DistributionNormalNode"),
          mean->cached_hash_value,
          stddev->cached_hash_value)},
      mean{mean},
      stddev{stddev} {}

void DistributionNormalNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

DistributionHalfNormalNode::DistributionHalfNormalNode(
    const ScalarNodep& stddev)
    : DistributionNode{hash_combine(
          std::hash<std::string>{}("DistributionHalfNormalNode"),
          stddev->cached_hash_value)},
      stddev{stddev} {}

void DistributionHalfNormalNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

DistributionBetaNode::DistributionBetaNode(
    const ScalarNodep& a,
    const ScalarNodep& b)
    : DistributionNode{hash_combine(
          std::hash<std::string>{}("DistributionBetaNode"),
          a->cached_hash_value,
          b->cached_hash_value)},
      a{a},
      b{b} {}

void DistributionBetaNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

DistributionBernoulliNode::DistributionBernoulliNode(const ScalarNodep& prob)
    : DistributionNode{hash_combine(
          std::hash<std::string>{}("DistributionBernoulliNode"),
          prob->cached_hash_value)},
      prob{prob} {}

void DistributionBernoulliNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

DistributionExponentialNode::DistributionExponentialNode(
    const ScalarNodep& rate)
    : DistributionNode{hash_combine(
          std::hash<std::string>{}("DistributionExponentialNode"),
          rate->cached_hash_value)},
      rate{rate} {}

void DistributionExponentialNode::accept(NodeVisitor& visitor) const {
  visitor.visit(this);
}

std::vector<Nodep> in_nodes(const Nodep& n) {
  return in_node_gatherer::gather(n);
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
  // equal nodes have equal hash codes and the same types
  if (l == nullptr || r == nullptr ||
      l->cached_hash_value != r->cached_hash_value ||
      typeid(*l) != typeid(*r)) {
    return false;
  }

  class EqualsTester : public NodeVisitor {
   public:
    const Node* other;
    const NodepIdentityEquals& node_equals;
    EqualsTester(const Node* other, const NodepIdentityEquals& node_equals)
        : other{other}, node_equals{node_equals} {}
    bool result;
    void visit(const ScalarConstantNode* node) override {
      auto other = static_cast<const ScalarConstantNode*>(this->other);
      result = node->constant_value == other->constant_value;
    }
    void visit(const ScalarVariableNode* node) override {
      auto other = static_cast<const ScalarVariableNode*>(this->other);
      result =
          node->identifier == other->identifier && node->name == other->name;
    }
    void visit(const ScalarSampleNode* node) override {
      auto other = static_cast<const ScalarSampleNode*>(this->other);
      result = node_equals(node->distribution, other->distribution);
    }
    void visit(const ScalarAddNode* node) override {
      auto other = static_cast<const ScalarAddNode*>(this->other);
      result = node_equals(node->left, other->left) &&
          node_equals(node->right, other->right);
    }
    void visit(const ScalarSubtractNode* node) override {
      auto other = static_cast<const ScalarSubtractNode*>(this->other);
      result = node_equals(node->left, other->left) &&
          node_equals(node->right, other->right);
    }
    void visit(const ScalarNegateNode* node) override {
      auto other = static_cast<const ScalarNegateNode*>(this->other);
      result = node_equals(node->x, other->x);
    }
    void visit(const ScalarMultiplyNode* node) override {
      auto other = static_cast<const ScalarMultiplyNode*>(this->other);
      result = node_equals(node->left, other->left) &&
          node_equals(node->right, other->right);
    }
    void visit(const ScalarDivideNode* node) override {
      auto other = static_cast<const ScalarDivideNode*>(this->other);
      result = node_equals(node->left, other->left) &&
          node_equals(node->right, other->right);
    }
    void visit(const ScalarPowNode* node) override {
      auto other = static_cast<const ScalarPowNode*>(this->other);
      result = node_equals(node->left, other->left) &&
          node_equals(node->right, other->right);
    }
    void visit(const ScalarExpNode* node) override {
      auto other = static_cast<const ScalarExpNode*>(this->other);
      result = node_equals(node->x, other->x);
    }
    void visit(const ScalarLogNode* node) override {
      auto other = static_cast<const ScalarLogNode*>(this->other);
      result = node_equals(node->x, other->x);
    }
    void visit(const ScalarAtanNode* node) override {
      auto other = static_cast<const ScalarAtanNode*>(this->other);
      result = node_equals(node->x, other->x);
    }
    void visit(const ScalarLgammaNode* node) override {
      auto other = static_cast<const ScalarLgammaNode*>(this->other);
      result = node_equals(node->x, other->x);
    }
    void visit(const ScalarPolygammaNode* node) override {
      auto other = static_cast<const ScalarPolygammaNode*>(this->other);
      result = node_equals(node->n, other->n) && node_equals(node->x, other->x);
    }
    void visit(const ScalarLog1pNode* node) override {
      auto other = static_cast<const ScalarLog1pNode*>(this->other);
      result = node_equals(node->x, other->x);
    }
    void visit(const ScalarIfEqualNode* node) override {
      auto other = static_cast<const ScalarIfEqualNode*>(this->other);
      result = node_equals(node->a, other->a) &&
          node_equals(node->b, other->b) && node_equals(node->c, other->c) &&
          node_equals(node->d, other->d);
    }
    void visit(const ScalarIfLessNode* node) override {
      auto other = static_cast<const ScalarIfLessNode*>(this->other);
      result = node_equals(node->a, other->a) &&
          node_equals(node->b, other->b) && node_equals(node->c, other->c) &&
          node_equals(node->d, other->d);
    }
    void visit(const DistributionNormalNode* node) override {
      auto other = static_cast<const DistributionNormalNode*>(this->other);
      result = node_equals(node->mean, other->mean) &&
          node_equals(node->stddev, other->stddev);
    }
    void visit(const DistributionHalfNormalNode* node) override {
      auto other = static_cast<const DistributionHalfNormalNode*>(this->other);
      result = node_equals(node->stddev, other->stddev);
    }
    void visit(const DistributionBetaNode* node) override {
      auto other = static_cast<const DistributionBetaNode*>(this->other);
      result = node_equals(node->a, other->a) && node_equals(node->b, other->b);
    }
    void visit(const DistributionBernoulliNode* node) override {
      auto other = static_cast<const DistributionBernoulliNode*>(this->other);
      result = node_equals(node->prob, other->prob);
    }
    void visit(const DistributionExponentialNode* node) override {
      auto other = static_cast<const DistributionExponentialNode*>(this->other);
      result = node_equals(node->rate, other->rate);
    }
  };

  EqualsTester et{r, *this};
  r->accept(et);
  return et.result;
}

void DefaultNodeVisitor::visit(const ScalarConstantNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const ScalarVariableNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const ScalarSampleNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const ScalarAddNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const ScalarSubtractNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const ScalarNegateNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const ScalarMultiplyNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const ScalarDivideNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const ScalarPowNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const ScalarExpNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const ScalarLogNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const ScalarAtanNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const ScalarLgammaNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const ScalarPolygammaNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const ScalarLog1pNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const ScalarIfEqualNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const ScalarIfLessNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const DistributionNormalNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const DistributionHalfNormalNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const DistributionBetaNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const DistributionBernoulliNode* node) {
  this->default_visit(node);
}
void DefaultNodeVisitor::visit(const DistributionExponentialNode* node) {
  this->default_visit(node);
}

void ScalarNodeVisitor::default_visit(const Node*) {
  throw std::logic_error("distribution passed to a scalar visitor");
}
void ScalarNodeVisitor::visit(const DistributionNormalNode* node) {
  default_visit(node);
}
void ScalarNodeVisitor::visit(const DistributionHalfNormalNode* node) {
  default_visit(node);
}
void ScalarNodeVisitor::visit(const DistributionBetaNode* node) {
  default_visit(node);
}
void ScalarNodeVisitor::visit(const DistributionBernoulliNode* node) {
  default_visit(node);
}
void ScalarNodeVisitor::visit(const DistributionExponentialNode* node) {
  default_visit(node);
}

void DistributionNodeVisitor::default_visit(const Node*) {
  throw std::logic_error("scalar passed to a distribution visitor");
}
void DistributionNodeVisitor::visit(const ScalarConstantNode* node) {
  default_visit(node);
}
void DistributionNodeVisitor::visit(const ScalarVariableNode* node) {
  default_visit(node);
}
void DistributionNodeVisitor::visit(const ScalarSampleNode* node) {
  default_visit(node);
}
void DistributionNodeVisitor::visit(const ScalarAddNode* node) {
  default_visit(node);
}
void DistributionNodeVisitor::visit(const ScalarSubtractNode* node) {
  default_visit(node);
}
void DistributionNodeVisitor::visit(const ScalarNegateNode* node) {
  default_visit(node);
}
void DistributionNodeVisitor::visit(const ScalarMultiplyNode* node) {
  default_visit(node);
}
void DistributionNodeVisitor::visit(const ScalarDivideNode* node) {
  default_visit(node);
}
void DistributionNodeVisitor::visit(const ScalarPowNode* node) {
  default_visit(node);
}
void DistributionNodeVisitor::visit(const ScalarExpNode* node) {
  default_visit(node);
}
void DistributionNodeVisitor::visit(const ScalarLogNode* node) {
  default_visit(node);
}
void DistributionNodeVisitor::visit(const ScalarAtanNode* node) {
  default_visit(node);
}
void DistributionNodeVisitor::visit(const ScalarLgammaNode* node) {
  default_visit(node);
}
void DistributionNodeVisitor::visit(const ScalarPolygammaNode* node) {
  default_visit(node);
}
void DistributionNodeVisitor::visit(const ScalarLog1pNode* node) {
  default_visit(node);
}
void DistributionNodeVisitor::visit(const ScalarIfEqualNode* node) {
  default_visit(node);
}
void DistributionNodeVisitor::visit(const ScalarIfLessNode* node) {
  default_visit(node);
}

} // namespace beanmachine::minibmg
