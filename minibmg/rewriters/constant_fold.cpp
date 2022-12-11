/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/rewriters/constant_fold.h"
#include <boost/math/special_functions/polygamma.hpp>
#include <cmath>
#include <memory>

namespace {

using namespace beanmachine::minibmg;

// A visitor that performs constant folding on a single node when its children
// are constants.  Otherwise returns the original node.
class ConstantFoldingVisitor : DefaultNodeVisitor {
 public:
  static Nodep constant_fold(Nodep node) {
    ConstantFoldingVisitor v;
    v.original = node;
    node->accept(v);
    if (v.rewritten == nullptr) {
      throw std::logic_error("missing node rewrite case");
    }

    return v.rewritten;
  }

 private:
  Nodep original;
  Nodep rewritten;

  explicit ConstantFoldingVisitor() {}

  void default_visit(const Node*) override {
    rewritten = original;
  }

  void visit(const ScalarAddNode* node) override {
    auto left_constant = downcast<ScalarConstantNode>(node->left);
    auto right_constant = downcast<ScalarConstantNode>(node->right);
    if (left_constant && right_constant) {
      rewritten = std::make_shared<const ScalarConstantNode>(
          left_constant->constant_value + right_constant->constant_value);
    } else {
      rewritten = original;
    }
  }

  void visit(const ScalarSubtractNode* node) override {
    auto left_constant = downcast<ScalarConstantNode>(node->left);
    auto right_constant = downcast<ScalarConstantNode>(node->right);
    if (left_constant && right_constant) {
      rewritten = std::make_shared<const ScalarConstantNode>(
          left_constant->constant_value - right_constant->constant_value);
    } else {
      rewritten = original;
    }
  }

  void visit(const ScalarNegateNode* node) override {
    auto x_constant = downcast<ScalarConstantNode>(node->x);
    if (x_constant) {
      rewritten = std::make_shared<const ScalarConstantNode>(
          -x_constant->constant_value);
    } else {
      rewritten = original;
    }
  }

  void visit(const ScalarMultiplyNode* node) override {
    auto left_constant = downcast<ScalarConstantNode>(node->left);
    auto right_constant = downcast<ScalarConstantNode>(node->right);
    if (left_constant && right_constant) {
      rewritten = std::make_shared<const ScalarConstantNode>(
          left_constant->constant_value * right_constant->constant_value);
    } else {
      rewritten = original;
    }
  }

  void visit(const ScalarDivideNode* node) override {
    auto left_constant = downcast<ScalarConstantNode>(node->left);
    auto right_constant = downcast<ScalarConstantNode>(node->right);
    if (left_constant && right_constant) {
      rewritten = std::make_shared<const ScalarConstantNode>(
          left_constant->constant_value / right_constant->constant_value);
    } else {
      rewritten = original;
    }
  }

  void visit(const ScalarPowNode* node) override {
    auto left_constant = downcast<ScalarConstantNode>(node->left);
    auto right_constant = downcast<ScalarConstantNode>(node->right);
    if (left_constant && right_constant) {
      auto k3 = std::pow(
          left_constant->constant_value, right_constant->constant_value);
      rewritten = std::make_shared<const ScalarConstantNode>(k3);
    } else {
      rewritten = original;
    }
  }

  void visit(const ScalarExpNode* node) override {
    auto x_constant = downcast<ScalarConstantNode>(node->x);
    if (x_constant) {
      rewritten = std::make_shared<const ScalarConstantNode>(
          std::exp(x_constant->constant_value));
    } else {
      rewritten = original;
    }
  }

  void visit(const ScalarLogNode* node) override {
    auto x_constant = downcast<ScalarConstantNode>(node->x);
    if (x_constant) {
      rewritten = std::make_shared<const ScalarConstantNode>(
          std::log(x_constant->constant_value));
    } else {
      rewritten = original;
    }
  }

  void visit(const ScalarAtanNode* node) override {
    auto x_constant = downcast<ScalarConstantNode>(node->x);
    if (x_constant) {
      rewritten = std::make_shared<const ScalarConstantNode>(
          std::atan(x_constant->constant_value));
    } else {
      rewritten = original;
    }
  }

  void visit(const ScalarLgammaNode* node) override {
    auto x_constant = downcast<ScalarConstantNode>(node->x);
    if (x_constant) {
      rewritten = std::make_shared<const ScalarConstantNode>(
          std::lgamma(x_constant->constant_value));
    } else {
      rewritten = original;
    }
  }

  void visit(const ScalarPolygammaNode* node) override {
    auto n_constant = downcast<ScalarConstantNode>(node->n);
    auto x_constant = downcast<ScalarConstantNode>(node->x);
    if (n_constant && x_constant) {
      auto value = boost::math::polygamma(
          n_constant->constant_value, x_constant->constant_value);
      rewritten = std::make_shared<const ScalarConstantNode>(value);
    } else {
      rewritten = original;
    }
  }

  void visit(const ScalarLog1pNode* node) override {
    auto x_constant = downcast<ScalarConstantNode>(node->x);
    if (x_constant) {
      rewritten = std::make_shared<const ScalarConstantNode>(
          std::log1p(x_constant->constant_value));
    } else {
      rewritten = original;
    }
  }

  void visit(const ScalarIfEqualNode* node) override {
    auto a_constant = downcast<ScalarConstantNode>(node->a);
    auto b_constant = downcast<ScalarConstantNode>(node->b);
    if (a_constant && b_constant) {
      rewritten = (a_constant->constant_value == b_constant->constant_value)
          ? node->c
          : node->d;
    } else {
      rewritten = original;
    }
  }

  void visit(const ScalarIfLessNode* node) override {
    auto a_constant = downcast<ScalarConstantNode>(node->a);
    auto b_constant = downcast<ScalarConstantNode>(node->b);
    if (a_constant && b_constant) {
      rewritten = (a_constant->constant_value < b_constant->constant_value)
          ? node->c
          : node->d;
    } else {
      rewritten = original;
    }
  }
};

} // namespace

namespace beanmachine::minibmg {

Nodep constant_fold(Nodep node) {
  return ConstantFoldingVisitor::constant_fold(node);
}
ScalarNodep constant_fold(ScalarNodep node) {
  return std::dynamic_pointer_cast<const ScalarNode>(
      ConstantFoldingVisitor::constant_fold(node));
}

} // namespace beanmachine::minibmg
