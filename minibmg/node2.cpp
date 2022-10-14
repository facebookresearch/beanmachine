/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/node2.h"
#include <boost/functional/hash.hpp>
#include <fmt/format.h>
#include <atomic>
#include <typeinfo>

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

} // namespace

namespace beanmachine::minibmg {

Node2::Node2(std::size_t cached_hash_value)
    : cached_hash_value{cached_hash_value} {}

Node2::~Node2() {}

ScalarNode2::ScalarNode2(std::size_t cached_hash_value)
    : Node2{cached_hash_value} {}

DistributionNode2::DistributionNode2(std::size_t cached_hash_value)
    : Node2{cached_hash_value} {}

std::string make_fresh_rvid() {
  static std::atomic<long> next_rvid = 1;
  return fmt::format("S{}", next_rvid.fetch_add(1));
}

ScalarConstantNode2::ScalarConstantNode2(double constant_value)
    : ScalarNode2{hash_combine(
          std::hash<std::string>{}("ScalarConstantNode2"),
          std::hash<double>{}(constant_value))} {}

void ScalarConstantNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

ScalarVariableNode2::ScalarVariableNode2(
    const std::string& name,
    const unsigned identifier)
    : ScalarNode2{hash_combine(
          std::hash<std::string>{}("ScalarVariableNode2"),
          std::hash<std::string>{}(name),
          std::hash<unsigned>{}(identifier))},
      name{name},
      identifier{identifier} {}

void ScalarVariableNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

ScalarSampleNode2::ScalarSampleNode2(
    const DistributionNode2p& distribution,
    const std::string& rvid)
    : ScalarNode2{hash_combine(
          std::hash<std::string>{}("ScalarSampleNode2"),
          distribution->cached_hash_value,
          std::hash<std::string>{}(rvid))},
      distribution{distribution},
      rvid{rvid} {}

void ScalarSampleNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

ScalarAddNode2::ScalarAddNode2(
    const ScalarNode2p& left,
    const ScalarNode2p& right)
    : ScalarNode2{hash_combine(
          std::hash<std::string>{}("ScalarAddNode2"),
          left->cached_hash_value,
          right->cached_hash_value)},
      left{left},
      right{right} {}

void ScalarAddNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

ScalarSubtractNode2::ScalarSubtractNode2(
    const ScalarNode2p& left,
    const ScalarNode2p& right)
    : ScalarNode2{hash_combine(
          std::hash<std::string>{}("ScalarSubtractNode2"),
          left->cached_hash_value,
          right->cached_hash_value)},
      left{left},
      right{right} {}

void ScalarSubtractNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

ScalarNegateNode2::ScalarNegateNode2(const ScalarNode2p& x)
    : ScalarNode2{hash_combine(
          std::hash<std::string>{}("ScalarNegateNode2"),
          x->cached_hash_value)},
      x{x} {}

void ScalarNegateNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

ScalarMultiplyNode2::ScalarMultiplyNode2(
    const ScalarNode2p& left,
    const ScalarNode2p& right)
    : ScalarNode2{hash_combine(
          std::hash<std::string>{}("ScalarMultiplyNode2"),
          left->cached_hash_value,
          right->cached_hash_value)},
      left{left},
      right{right} {}

void ScalarMultiplyNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

ScalarDivideNode2::ScalarDivideNode2(
    const ScalarNode2p& left,
    const ScalarNode2p& right)
    : ScalarNode2{hash_combine(
          std::hash<std::string>{}("ScalarDivideNode2"),
          left->cached_hash_value,
          right->cached_hash_value)},
      left{left},
      right{right} {}

void ScalarDivideNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

ScalarPowNode2::ScalarPowNode2(
    const ScalarNode2p& left,
    const ScalarNode2p& right)
    : ScalarNode2{hash_combine(
          std::hash<std::string>{}("ScalarPowNode2"),
          left->cached_hash_value,
          right->cached_hash_value)},
      left{left},
      right{right} {}

void ScalarPowNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

ScalarExpNode2::ScalarExpNode2(const ScalarNode2p& x)
    : ScalarNode2{hash_combine(
          std::hash<std::string>{}("ScalarExpNode2"),
          x->cached_hash_value)},
      x{x} {}

void ScalarExpNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

ScalarLogNode2::ScalarLogNode2(const ScalarNode2p& x)
    : ScalarNode2{hash_combine(
          std::hash<std::string>{}("ScalarLogNode2"),
          x->cached_hash_value)},
      x{x} {}

void ScalarLogNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

ScalarAtanNode2::ScalarAtanNode2(const ScalarNode2p& x)
    : ScalarNode2{hash_combine(
          std::hash<std::string>{}("ScalarAtanNode2"),
          x->cached_hash_value)},
      x{x} {}

void ScalarAtanNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

ScalarLgammaNode2::ScalarLgammaNode2(const ScalarNode2p& x)
    : ScalarNode2{hash_combine(
          std::hash<std::string>{}("ScalarLgammaNode2"),
          x->cached_hash_value)},
      x{x} {}

void ScalarLgammaNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

ScalarPolygammaNode2::ScalarPolygammaNode2(
    const ScalarNode2p& n,
    const ScalarNode2p& x)
    : ScalarNode2{hash_combine(
          std::hash<std::string>{}("ScalarPolygammaNode2"),
          n->cached_hash_value,
          x->cached_hash_value)},
      n{n},
      x{x} {}

void ScalarPolygammaNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

ScalarIfEqualNode2::ScalarIfEqualNode2(
    const ScalarNode2p& a,
    const ScalarNode2p& b,
    const ScalarNode2p& c,
    const ScalarNode2p& d)
    : ScalarNode2{hash_combine(
          {std::hash<std::string>{}("ScalarIfEqualNode2"),
           a->cached_hash_value,
           b->cached_hash_value,
           c->cached_hash_value,
           d->cached_hash_value})},
      a{a},
      b{b},
      c{c},
      d{d} {}

void ScalarIfEqualNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

ScalarIfLessNode2::ScalarIfLessNode2(
    const ScalarNode2p& a,
    const ScalarNode2p& b,
    const ScalarNode2p& c,
    const ScalarNode2p& d)
    : ScalarNode2{hash_combine(
          {std::hash<std::string>{}("ScalarIfLessNode2"),
           a->cached_hash_value,
           b->cached_hash_value,
           c->cached_hash_value,
           d->cached_hash_value})},
      a{a},
      b{b},
      c{c},
      d{d} {}

void ScalarIfLessNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

DistributionNormalNode2::DistributionNormalNode2(
    const ScalarNode2p& mean,
    const ScalarNode2p& stddev)
    : DistributionNode2{hash_combine(
          std::hash<std::string>{}("DistributionNormalNode2"),
          mean->cached_hash_value,
          stddev->cached_hash_value)},
      mean{mean},
      stddev{stddev} {}

void DistributionNormalNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

DistributionHalfNormalNode2::DistributionHalfNormalNode2(
    const ScalarNode2p& stddev)
    : DistributionNode2{hash_combine(
          std::hash<std::string>{}("DistributionHalfNormalNode2"),
          stddev->cached_hash_value)},
      stddev{stddev} {}

void DistributionHalfNormalNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

DistributionBetaNode2::DistributionBetaNode2(
    const ScalarNode2p& a,
    const ScalarNode2p& b)
    : DistributionNode2{hash_combine(
          std::hash<std::string>{}("DistributionBetaNode2"),
          a->cached_hash_value,
          b->cached_hash_value)},
      a{a},
      b{b} {}

void DistributionBetaNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

DistributionBernoulliNode2::DistributionBernoulliNode2(const ScalarNode2p& prob)
    : DistributionNode2{hash_combine(
          std::hash<std::string>{}("DistributionBernoulliNode2"),
          prob->cached_hash_value)},
      prob{prob} {}

void DistributionBernoulliNode2::accept(Node2Visitor& visitor) const {
  visitor.visit(this);
}

std::vector<Node2p> in_nodes(const Node2p& n) {
  class in_node_gatherer : public Node2Visitor {
   public:
    std::vector<Node2p> result;
    void visit(const ScalarConstantNode2*) override {
      result = {};
    }
    void visit(const ScalarVariableNode2*) override {
      result = {};
    }
    void visit(const ScalarSampleNode2* n) override {
      result = {n->distribution};
    }
    void visit(const ScalarAddNode2* n) override {
      result = {n->left, n->right};
    }
    void visit(const ScalarSubtractNode2* n) override {
      result = {n->left, n->right};
    }
    void visit(const ScalarNegateNode2* n) override {
      result = {n->x};
    }
    void visit(const ScalarMultiplyNode2* n) override {
      result = {n->left, n->right};
    }
    void visit(const ScalarDivideNode2* n) override {
      result = {n->left, n->right};
    }
    void visit(const ScalarPowNode2* n) override {
      result = {n->left, n->right};
    }
    void visit(const ScalarExpNode2* n) override {
      result = {n->x};
    }
    void visit(const ScalarLogNode2* n) override {
      result = {n->x};
    }
    void visit(const ScalarAtanNode2* n) override {
      result = {n->x};
    }
    void visit(const ScalarLgammaNode2* n) override {
      result = {n->x};
    }
    void visit(const ScalarPolygammaNode2* n) override {
      result = {n->n, n->x};
    }
    void visit(const ScalarIfEqualNode2* n) override {
      result = {n->a, n->b, n->c, n->d};
    }
    void visit(const ScalarIfLessNode2* n) override {
      result = {n->a, n->b, n->c, n->d};
    }
    void visit(const DistributionNormalNode2* n) override {
      result = {n->mean, n->stddev};
    }
    void visit(const DistributionHalfNormalNode2* n) override {
      result = {n->stddev};
    }
    void visit(const DistributionBetaNode2* n) override {
      result = {n->a, n->b};
    }
    void visit(const DistributionBernoulliNode2* n) override {
      result = {n->prob};
    }
  };
  in_node_gatherer gatherer;
  n->accept(gatherer);
  return std::move(gatherer.result);
}

std::size_t Node2pIdentityHash::operator()(
    beanmachine::minibmg::Node2p const& p) const noexcept {
  return p->cached_hash_value;
}

bool Node2pIdentityEquals::operator()(
    const beanmachine::minibmg::Node2p& lhs,
    const beanmachine::minibmg::Node2p& rhs) const noexcept {
  const Node2* l = lhs.get();
  const Node2* r = rhs.get();
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

  class EqualsTester : public Node2Visitor {
   public:
    const Node2* other;
    const Node2pIdentityEquals& node_equals;
    EqualsTester(const Node2* other, const Node2pIdentityEquals& node_equals)
        : other{other}, node_equals{node_equals} {}
    bool result;
    void visit(const ScalarConstantNode2* node) override {
      auto other = static_cast<const ScalarConstantNode2*>(this->other);
      result = node->constant_value == other->constant_value;
    }
    void visit(const ScalarVariableNode2* node) override {
      auto other = static_cast<const ScalarVariableNode2*>(this->other);
      result =
          node->identifier == other->identifier && node->name == other->name;
    }
    void visit(const ScalarSampleNode2* node) override {
      auto other = static_cast<const ScalarSampleNode2*>(this->other);
      result = node_equals(node->distribution, other->distribution);
    }
    void visit(const ScalarAddNode2* node) override {
      auto other = static_cast<const ScalarAddNode2*>(this->other);
      result = node_equals(node->left, other->left) &&
          node_equals(node->right, other->right);
    }
    void visit(const ScalarSubtractNode2* node) override {
      auto other = static_cast<const ScalarSubtractNode2*>(this->other);
      result = node_equals(node->left, other->left) &&
          node_equals(node->right, other->right);
    }
    void visit(const ScalarNegateNode2* node) override {
      auto other = static_cast<const ScalarNegateNode2*>(this->other);
      result = node_equals(node->x, other->x);
    }
    void visit(const ScalarMultiplyNode2* node) override {
      auto other = static_cast<const ScalarMultiplyNode2*>(this->other);
      result = node_equals(node->left, other->left) &&
          node_equals(node->right, other->right);
    }
    void visit(const ScalarDivideNode2* node) override {
      auto other = static_cast<const ScalarDivideNode2*>(this->other);
      result = node_equals(node->left, other->left) &&
          node_equals(node->right, other->right);
    }
    void visit(const ScalarPowNode2* node) override {
      auto other = static_cast<const ScalarPowNode2*>(this->other);
      result = node_equals(node->left, other->left) &&
          node_equals(node->right, other->right);
    }
    void visit(const ScalarExpNode2* node) override {
      auto other = static_cast<const ScalarExpNode2*>(this->other);
      result = node_equals(node->x, other->x);
    }
    void visit(const ScalarLogNode2* node) override {
      auto other = static_cast<const ScalarLogNode2*>(this->other);
      result = node_equals(node->x, other->x);
    }
    void visit(const ScalarAtanNode2* node) override {
      auto other = static_cast<const ScalarAtanNode2*>(this->other);
      result = node_equals(node->x, other->x);
    }
    void visit(const ScalarLgammaNode2* node) override {
      auto other = static_cast<const ScalarLgammaNode2*>(this->other);
      result = node_equals(node->x, other->x);
    }
    void visit(const ScalarPolygammaNode2* node) override {
      auto other = static_cast<const ScalarPolygammaNode2*>(this->other);
      result = node_equals(node->n, other->n) && node_equals(node->x, other->x);
    }
    void visit(const ScalarIfEqualNode2* node) override {
      auto other = static_cast<const ScalarIfEqualNode2*>(this->other);
      result = node_equals(node->a, other->a) &&
          node_equals(node->b, other->b) && node_equals(node->c, other->c) &&
          node_equals(node->d, other->d);
    }
    void visit(const ScalarIfLessNode2* node) override {
      auto other = static_cast<const ScalarIfLessNode2*>(this->other);
      result = node_equals(node->a, other->a) &&
          node_equals(node->b, other->b) && node_equals(node->c, other->c) &&
          node_equals(node->d, other->d);
    }
    void visit(const DistributionNormalNode2* node) override {
      auto other = static_cast<const DistributionNormalNode2*>(this->other);
      result = node_equals(node->mean, other->mean) &&
          node_equals(node->stddev, other->stddev);
    }
    void visit(const DistributionHalfNormalNode2* node) override {
      auto other = static_cast<const DistributionHalfNormalNode2*>(this->other);
      result = node_equals(node->stddev, other->stddev);
    }
    void visit(const DistributionBetaNode2* node) override {
      auto other = static_cast<const DistributionBetaNode2*>(this->other);
      result = node_equals(node->a, other->a) && node_equals(node->b, other->b);
    }
    void visit(const DistributionBernoulliNode2* node) override {
      auto other = static_cast<const DistributionBernoulliNode2*>(this->other);
      result = node_equals(node->prob, other->prob);
    }
  };

  EqualsTester et{r, *this};
  r->accept(et);
  return et.result;
}

void DefaultNode2Visitor::visit(const ScalarConstantNode2* node) {
  this->default_visit(node);
}
void DefaultNode2Visitor::visit(const ScalarVariableNode2* node) {
  this->default_visit(node);
}
void DefaultNode2Visitor::visit(const ScalarSampleNode2* node) {
  this->default_visit(node);
}
void DefaultNode2Visitor::visit(const ScalarAddNode2* node) {
  this->default_visit(node);
}
void DefaultNode2Visitor::visit(const ScalarSubtractNode2* node) {
  this->default_visit(node);
}
void DefaultNode2Visitor::visit(const ScalarNegateNode2* node) {
  this->default_visit(node);
}
void DefaultNode2Visitor::visit(const ScalarMultiplyNode2* node) {
  this->default_visit(node);
}
void DefaultNode2Visitor::visit(const ScalarDivideNode2* node) {
  this->default_visit(node);
}
void DefaultNode2Visitor::visit(const ScalarPowNode2* node) {
  this->default_visit(node);
}
void DefaultNode2Visitor::visit(const ScalarExpNode2* node) {
  this->default_visit(node);
}
void DefaultNode2Visitor::visit(const ScalarLogNode2* node) {
  this->default_visit(node);
}
void DefaultNode2Visitor::visit(const ScalarAtanNode2* node) {
  this->default_visit(node);
}
void DefaultNode2Visitor::visit(const ScalarLgammaNode2* node) {
  this->default_visit(node);
}
void DefaultNode2Visitor::visit(const ScalarPolygammaNode2* node) {
  this->default_visit(node);
}
void DefaultNode2Visitor::visit(const ScalarIfEqualNode2* node) {
  this->default_visit(node);
}
void DefaultNode2Visitor::visit(const ScalarIfLessNode2* node) {
  this->default_visit(node);
}
void DefaultNode2Visitor::visit(const DistributionNormalNode2* node) {
  this->default_visit(node);
}
void DefaultNode2Visitor::visit(const DistributionHalfNormalNode2* node) {
  this->default_visit(node);
}
void DefaultNode2Visitor::visit(const DistributionBetaNode2* node) {
  this->default_visit(node);
}
void DefaultNode2Visitor::visit(const DistributionBernoulliNode2* node) {
  this->default_visit(node);
}

void ScalarNode2Visitor::default_visit(const Node2*) {
  throw std::logic_error("distribution passed to a scalar visitor");
}
void ScalarNode2Visitor::visit(const DistributionNormalNode2* node) {
  default_visit(node);
}
void ScalarNode2Visitor::visit(const DistributionHalfNormalNode2* node) {
  default_visit(node);
}
void ScalarNode2Visitor::visit(const DistributionBetaNode2* node) {
  default_visit(node);
}
void ScalarNode2Visitor::visit(const DistributionBernoulliNode2* node) {
  default_visit(node);
}

void DistributionNode2Visitor::default_visit(const Node2*) {
  throw std::logic_error("scalar passed to a distribution visitor");
}
void DistributionNode2Visitor::visit(const ScalarConstantNode2* node) {
  default_visit(node);
}
void DistributionNode2Visitor::visit(const ScalarVariableNode2* node) {
  default_visit(node);
}
void DistributionNode2Visitor::visit(const ScalarSampleNode2* node) {
  default_visit(node);
}
void DistributionNode2Visitor::visit(const ScalarAddNode2* node) {
  default_visit(node);
}
void DistributionNode2Visitor::visit(const ScalarSubtractNode2* node) {
  default_visit(node);
}
void DistributionNode2Visitor::visit(const ScalarNegateNode2* node) {
  default_visit(node);
}
void DistributionNode2Visitor::visit(const ScalarMultiplyNode2* node) {
  default_visit(node);
}
void DistributionNode2Visitor::visit(const ScalarDivideNode2* node) {
  default_visit(node);
}
void DistributionNode2Visitor::visit(const ScalarPowNode2* node) {
  default_visit(node);
}
void DistributionNode2Visitor::visit(const ScalarExpNode2* node) {
  default_visit(node);
}
void DistributionNode2Visitor::visit(const ScalarLogNode2* node) {
  default_visit(node);
}
void DistributionNode2Visitor::visit(const ScalarAtanNode2* node) {
  default_visit(node);
}
void DistributionNode2Visitor::visit(const ScalarLgammaNode2* node) {
  default_visit(node);
}
void DistributionNode2Visitor::visit(const ScalarPolygammaNode2* node) {
  default_visit(node);
}
void DistributionNode2Visitor::visit(const ScalarIfEqualNode2* node) {
  default_visit(node);
}
void DistributionNode2Visitor::visit(const ScalarIfLessNode2* node) {
  default_visit(node);
}

} // namespace beanmachine::minibmg
