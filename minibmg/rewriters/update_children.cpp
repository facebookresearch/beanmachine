/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/rewriters/update_children.h"
#include <memory>

namespace {

using namespace beanmachine::minibmg;

// A visitor that rewrites a single node (at a time) by replacing children
// according to a given map.
class UpdateChildrenVisitor : NodeVisitor {
 public:
  static Nodep update_children(
      Nodep node,
      const std::unordered_map<Nodep, Nodep>& map) {
    auto visitor = UpdateChildrenVisitor{map};
    visitor.original = node;
    node->accept(visitor);
    if (visitor.rewritten == nullptr) {
      throw std::logic_error("missing node rewrite case");
    }

    return visitor.rewritten;
  }

 private:
  const std::unordered_map<Nodep, Nodep>& map;
  Nodep original;
  Nodep rewritten;

  explicit UpdateChildrenVisitor(const std::unordered_map<Nodep, Nodep>& map)
      : map{map} {}

  ScalarNodep mapped(const ScalarNodep& x) {
    auto r = map.at(x);
    return downcast<ScalarNode>(r);
  }

  DistributionNodep mapped(const DistributionNodep& x) {
    auto r = map.at(x);
    return downcast<DistributionNode>(r);
  }

  void visit(const ScalarConstantNode*) override {
    rewritten = original;
  }

  void visit(const ScalarVariableNode*) override {
    rewritten = original;
  }

  void visit(const ScalarSampleNode* node) override {
    auto d = mapped(node->distribution);
    if (node->distribution != d) {
      rewritten = std::make_shared<const ScalarSampleNode>(d, node->rvid);
    } else {
      rewritten = original;
    }
  }

  void visit(const ScalarAddNode* node) override {
    auto left = mapped(node->left);
    auto right = mapped(node->right);
    if (left != node->left || right != node->right) {
      rewritten = std::make_shared<const ScalarAddNode>(left, right);
    } else {
      rewritten = original;
    }
  }

  void visit(const ScalarSubtractNode* node) override {
    auto left = mapped(node->left);
    auto right = mapped(node->right);
    if (left != node->left || right != node->right) {
      rewritten = std::make_shared<const ScalarSubtractNode>(left, right);
    } else {
      rewritten = original;
    }
  }

  void visit(const ScalarNegateNode* node) override {
    auto x = mapped(node->x);
    if (x != node->x) {
      rewritten = std::make_shared<const ScalarNegateNode>(x);
    } else {
      rewritten = original;
    }
  }

  void visit(const ScalarMultiplyNode* node) override {
    auto left = mapped(node->left);
    auto right = mapped(node->right);
    if (left != node->left || right != node->right) {
      rewritten = std::make_shared<const ScalarMultiplyNode>(left, right);
    } else {
      rewritten = original;
    }
  }

  void visit(const ScalarDivideNode* node) override {
    auto left = mapped(node->left);
    auto right = mapped(node->right);
    if (left != node->left || right != node->right) {
      rewritten = std::make_shared<const ScalarDivideNode>(left, right);
    } else {
      rewritten = original;
    }
  }

  void visit(const ScalarPowNode* node) override {
    auto left = mapped(node->left);
    auto right = mapped(node->right);
    if (left == node->left && right == node->right) {
      rewritten = original;
    } else {
      rewritten = std::make_shared<const ScalarPowNode>(left, right);
    }
  }

  void visit(const ScalarExpNode* node) override {
    auto x = mapped(node->x);
    if (x == node->x) {
      rewritten = original;
    } else {
      rewritten = std::make_shared<const ScalarExpNode>(x);
    }
  }

  void visit(const ScalarLogNode* node) override {
    auto x = mapped(node->x);
    if (x == node->x) {
      rewritten = original;
    } else {
      rewritten = std::make_shared<const ScalarLogNode>(x);
    }
  }

  void visit(const ScalarAtanNode* node) override {
    auto x = mapped(node->x);
    if (x == node->x) {
      rewritten = original;
    } else {
      rewritten = std::make_shared<const ScalarAtanNode>(x);
    }
  }

  void visit(const ScalarLgammaNode* node) override {
    auto x = mapped(node->x);
    if (x == node->x) {
      rewritten = original;
    } else {
      rewritten = std::make_shared<const ScalarLgammaNode>(x);
    }
  }

  void visit(const ScalarPolygammaNode* node) override {
    auto n = mapped(node->n);
    auto x = mapped(node->x);
    if (n == node->n && x == node->x) {
      rewritten = original;
    } else {
      rewritten = std::make_shared<const ScalarPolygammaNode>(n, x);
    }
  }

  void visit(const ScalarLog1pNode* node) override {
    auto x = mapped(node->x);
    if (x == node->x) {
      rewritten = original;
    } else {
      rewritten = std::make_shared<const ScalarLog1pNode>(x);
    }
  }

  void visit(const ScalarIfEqualNode* node) override {
    auto a = mapped(node->a);
    auto b = mapped(node->b);
    auto c = mapped(node->c);
    auto d = mapped(node->d);
    if (a == node->a && b == node->b && c == node->c && d == node->d) {
      rewritten = original;
    } else {
      rewritten = std::make_shared<const ScalarIfEqualNode>(a, b, c, d);
    }
  }

  void visit(const ScalarIfLessNode* node) override {
    auto a = mapped(node->a);
    auto b = mapped(node->b);
    auto c = mapped(node->c);
    auto d = mapped(node->d);
    if (a == node->a && b == node->b && c == node->c && d == node->d) {
      rewritten = original;
    } else {
      rewritten = std::make_shared<const ScalarIfLessNode>(a, b, c, d);
    }
  }

  void visit(const DistributionNormalNode* node) override {
    auto mean = mapped(node->mean);
    auto stddev = mapped(node->stddev);
    if (mean == node->mean && stddev == node->stddev) {
      rewritten = original;
    } else {
      rewritten = std::make_shared<DistributionNormalNode>(mean, stddev);
    }
  }

  void visit(const DistributionHalfNormalNode* node) override {
    auto stddev = mapped(node->stddev);
    if (stddev == node->stddev) {
      rewritten = original;
    } else {
      rewritten = std::make_shared<DistributionHalfNormalNode>(stddev);
    }
  }

  void visit(const DistributionBetaNode* node) override {
    auto a = mapped(node->a);
    auto b = mapped(node->b);
    if (a == node->a && b == node->b) {
      rewritten = original;
    } else {
      rewritten = std::make_shared<DistributionBetaNode>(a, b);
    }
  }

  void visit(const DistributionBernoulliNode* node) override {
    auto prob = mapped(node->prob);
    if (prob == node->prob) {
      rewritten = original;
    } else {
      rewritten = std::make_shared<DistributionBernoulliNode>(prob);
    }
  }

  void visit(const DistributionExponentialNode* node) override {
    auto rate = mapped(node->rate);
    if (rate == node->rate) {
      rewritten = original;
    } else {
      rewritten = std::make_shared<DistributionExponentialNode>(rate);
    }
  }
};

} // namespace

namespace beanmachine::minibmg {

Nodep update_children(
    const Nodep& node,
    const std::unordered_map<Nodep, Nodep>& map) {
  return UpdateChildrenVisitor::update_children(node, map);
}
ScalarNodep update_children(
    const ScalarNodep& node,
    const std::unordered_map<Nodep, Nodep>& map) {
  Nodep n = node;
  return downcast<ScalarNode>(update_children(n, map));
}

} // namespace beanmachine::minibmg
