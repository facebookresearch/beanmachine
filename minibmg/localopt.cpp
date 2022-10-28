/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/localopt.h"
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include "beanmachine/minibmg/node.h"
#include "beanmachine/minibmg/topological.h"
#include "node.h"

namespace {

using namespace beanmachine::minibmg;

NodepValueEquals same{};

// This is a temporary hack to perform some local optimizations on the a graph
// node. Ultimately, these should be organized into a rewriter based on tree
// automata, which will make the rewriters much easier to maintain and much
// faster.  For now we hand-implement a few rules by brute force.  The one-line
// comment before each transformation shows what the rule would look like in a
// hypothetical rewriting system.
class RewriteOneVisitor : NodeVisitor {
 private:
  NodeNodeValueMap& map;
  Nodep original;
  Nodep rewritten;

 public:
  explicit RewriteOneVisitor(NodeNodeValueMap& map) : map{map} {}
  Nodep rewrite_one(const Nodep& node) {
    if (auto found = map.find(node); found != map.end()) {
      // A semantically equivalent node was already rewritten.
      return found->second;
    }

    rewritten = nullptr;
    original = node;
    node->accept(*this);
    if (rewritten == nullptr) {
      throw std::logic_error("missing node rewrite case");
    }

    return rewritten;
  }

  ScalarNodep rewrite_scalar_node(const ScalarNodep& node) {
    return std::dynamic_pointer_cast<const ScalarNode>(rewrite_node(node));
  }

  // The following method may be useful in debugging the problematic case
  // that the rewrite_one method returns nested nodes that it does not
  // place in the map. It is commented out in normal use, but when the
  // optimizer throws an exception because a node is not in the map, this
  // will likely be helpful in finding the problem.
  bool check_children(const Nodep& node) {
    for (auto in : in_nodes(node)) {
      if (!map.contains(in) || map.at(in) == nullptr) {
        return false;
      }
    }
    return true;
  }

  // Call the rewriter repeatedly on a node until a fixed-point is reached, and
  // then place the result in the node-value-based map.
  Nodep rewrite_node(const Nodep& node) {
    // check_children(node, map);
    Nodep rewritten = node;
    while (true) {
      const Nodep n = rewrite_one(rewritten);
      if (same(n, rewritten)) {
        rewritten = n;
        break;
      }
      if (n == nullptr) {
        throw std::logic_error("rewriter should not return nullptr");
      }
      rewritten = n;
    }

    if (auto found = map.find(node);
        found == map.end() || !same(rewritten, found->second)) {
      if (rewritten == nullptr) {
        throw std::logic_error("rewriter should not return nullptr");
      }
      map.insert(node, rewritten);
    }

    map.insert(rewritten, rewritten);
    return rewritten;
  }

 private:
  void visit(const ScalarConstantNode*) override {
    rewritten = original;
  }

  void visit(const ScalarVariableNode*) override {
    rewritten = original;
  }

  void visit(const ScalarSampleNode* node) override {
    auto d = map.at(node->distribution);
    if (node->distribution != d) {
      rewritten = std::make_shared<ScalarSampleNode>(d, node->rvid);
    }

    else {
      rewritten = original;
    }
  }

  void visit(const ScalarAddNode* node) override {
    auto left = map.at(node->left);
    auto right = map.at(node->right);
    auto left_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode>(left);
    auto right_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode>(right);
    auto left_product =
        std::dynamic_pointer_cast<const ScalarMultiplyNode>(left);

    // {k1 + k2, k3}, // constant fold
    if (left_constant && right_constant) {
      rewritten = std::make_shared<ScalarConstantNode>(
          left_constant->constant_value + right_constant->constant_value);
    }

    // {0 + x, x},
    else if (left_constant && left_constant->constant_value == 0) {
      rewritten = right;
    }

    // {x + 0, x},
    else if (right_constant && right_constant->constant_value == 0) {
      rewritten = left;
    }

    // {x + x, 2 * x},
    else if (same(left, right)) {
      ScalarNodep two =
          rewrite_scalar_node(std::make_shared<ScalarConstantNode>(2));
      rewritten = std::make_shared<ScalarMultiplyNode>(two, left);
    }

    // {y * x + x, (1 + y) * x}
    if (rewritten == nullptr) {
      auto left_product =
          std::dynamic_pointer_cast<const ScalarMultiplyNode>(left);
      if (left_product && same(left_product->right, right)) {
        ScalarNodep one =
            rewrite_scalar_node(std::make_shared<ScalarConstantNode>(1));
        auto new_left = rewrite_scalar_node(
            std::make_shared<ScalarAddNode>(one, left_product->left));
        rewritten = std::make_shared<ScalarMultiplyNode>(new_left, right);
      }
    }

    // {z + x + x, z + (2 * x)}
    if (rewritten == nullptr) {
      if (auto left_sum =
              std::dynamic_pointer_cast<const ScalarAddNode>(left)) {
        if (same(left_sum->right, right)) {
          ScalarNodep two =
              rewrite_scalar_node(std::make_shared<ScalarConstantNode>(2));
          auto new_left = left_sum->left;
          auto new_right = rewrite_scalar_node(
              std::make_shared<ScalarMultiplyNode>(two, right));
          rewritten = std::make_shared<ScalarAddNode>(new_left, new_right);
        }
      }
    }

    // {(z + (y * x)) + x, z + (1 + y) * x}
    if (rewritten == nullptr) {
      if (auto left_sum =
              std::dynamic_pointer_cast<const ScalarAddNode>(left)) {
        auto left_right_product =
            std::dynamic_pointer_cast<const ScalarMultiplyNode>(
                left_sum->right);
        if (left_right_product && same(left_right_product, right)) {
          ScalarNodep one =
              rewrite_scalar_node(std::make_shared<ScalarConstantNode>(1));
          auto one_plus_y = rewrite_scalar_node(
              std::make_shared<ScalarAddNode>(one, left_right_product->left));
          auto new_left = left_sum->left;
          auto new_right =
              rewrite_scalar_node(std::make_shared<ScalarMultiplyNode>(
                  one_plus_y, left_right_product->right));
          rewritten = rewrite_scalar_node(
              std::make_shared<ScalarAddNode>(new_left, new_right));
        }
      }
    }

    if (rewritten == nullptr) {
      if (left != node->left || right != node->right) {
        rewritten = std::make_shared<ScalarAddNode>(left, right);
      } else {
        rewritten = original;
      }
    }
  }

  void visit(const ScalarSubtractNode* node) override {
    auto left = map.at(node->left);
    auto right = map.at(node->right);
    auto left_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode>(left);
    auto right_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode>(right);

    // {k1 - k2, k3}, // constant fold
    if (left_constant && right_constant) {
      rewritten = std::make_shared<ScalarConstantNode>(
          left_constant->constant_value - right_constant->constant_value);
    }

    // {0 - x, -x},
    else if (left_constant && left_constant->constant_value == 0) {
      rewritten = std::make_shared<ScalarNegateNode>(right);
    }

    // {x - 0, x},
    else if (right_constant && right_constant->constant_value == 0) {
      rewritten = left;
    }

    // {x - x, 0},
    else if (same(left, right)) {
      rewritten = std::make_shared<ScalarConstantNode>(0);
    }

    else if (left != node->left || right != node->right) {
      rewritten = std::make_shared<ScalarSubtractNode>(left, right);
    }

    else {
      rewritten = original;
    }
  }

  void visit(const ScalarNegateNode* node) override {
    auto x = map.at(node->x);
    auto x_constant = std::dynamic_pointer_cast<const ScalarConstantNode>(x);

    // {-k, k3}, // constant fold
    if (x_constant) {
      rewritten =
          std::make_shared<ScalarConstantNode>(-x_constant->constant_value);
    }

    // {--x, x},
    else if (
        auto x_negate = std::dynamic_pointer_cast<const ScalarNegateNode>(x)) {
      rewritten = x_negate->x;
    }

    else if (x != node->x) {
      rewritten = std::make_shared<ScalarNegateNode>(x);
    }

    else {
      rewritten = original;
    }
  }

  void visit(const ScalarMultiplyNode* node) override {
    auto left = map.at(node->left);
    auto right = map.at(node->right);
    auto left_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode>(left);
    auto right_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode>(right);

    // {k1 * k2, k3}, // constant fold
    if (left_constant && right_constant) {
      rewritten = std::make_shared<ScalarConstantNode>(
          left_constant->constant_value * right_constant->constant_value);
    }

    // {0 * x, 0},
    else if (left_constant && left_constant->constant_value == 0) {
      rewritten = left;
    }

    // {-1 * x, -x},
    else if (left_constant && left_constant->constant_value == -1) {
      rewritten = std::make_shared<ScalarNegateNode>(right);
    }

    // {1 * x, x},
    else if (left_constant && left_constant->constant_value == 1) {
      rewritten = right;
    }

    // {x * 0, 0},
    else if (right_constant && right_constant->constant_value == 0) {
      rewritten = right;
    }

    // {x * 1, x},
    else if (right_constant && right_constant->constant_value == 1) {
      rewritten = left;
    }

    // {k1 * (k2 * x), k3 * x },
    else if (auto right_multiply =
                 std::dynamic_pointer_cast<const ScalarMultiplyNode>(right);
             left_constant && right_multiply) {
      if (auto right_left_constant =
              std::dynamic_pointer_cast<const ScalarConstantNode>(
                  right_multiply->left);
          right_left_constant) {
        auto k3 = rewrite_scalar_node(std::make_shared<ScalarConstantNode>(
            left_constant->constant_value *
            right_left_constant->constant_value));
        rewritten =
            std::make_shared<ScalarMultiplyNode>(k3, right_multiply->right);
      }
    }

    if (rewritten == nullptr) {
      if (left != node->left || right != node->right) {
        rewritten = std::make_shared<ScalarMultiplyNode>(left, right);
      }

      else {
        rewritten = original;
      }
    }
  }

  void visit(const ScalarDivideNode* node) override {
    auto left = map.at(node->left);
    auto right = map.at(node->right);
    auto left_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode>(left);
    auto right_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode>(right);

    // {k1 / k2, k3}, // constant fold
    if (left_constant && right_constant) {
      rewritten = std::make_shared<ScalarConstantNode>(
          left_constant->constant_value / right_constant->constant_value);
    }

    // {0 / x, 0},
    else if (left_constant && left_constant->constant_value == 0) {
      rewritten = left;
    }

    // {x / k, (1/k) * x},
    else if (right_constant) {
      auto k3 = rewrite_scalar_node(std::make_shared<ScalarConstantNode>(
          1 / right_constant->constant_value));
      rewritten = std::make_shared<ScalarMultiplyNode>(k3, left);
    }

    // {y / (y / x), x}
    else if (
        auto right_divide =
            std::dynamic_pointer_cast<const ScalarDivideNode>(right)) {
      if (same(left, right_divide->left)) {
        rewritten = right_divide->right;
      }
    }

    if (rewritten == nullptr) {
      if (left != node->left || right != node->right) {
        rewritten = std::make_shared<ScalarDivideNode>(left, right);
      } else {
        rewritten = original;
      }
    }
  }

  void visit(const ScalarPowNode* node) override {
    auto left = map.at(node->left);
    auto right = map.at(node->right);
    auto left_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode>(left);
    auto right_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode>(right);

    // {pow(k1, k2), k3}, // constant fold
    if (left_constant && right_constant) {
      auto k3 = std::pow(
          left_constant->constant_value, right_constant->constant_value);
      rewritten = std::make_shared<ScalarConstantNode>(k3);
    }

    // {pow(x, 1), x},
    else if (right_constant && right_constant->constant_value == 1) {
      rewritten = left;
    }

    else if (left == node->left && right == node->right) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<ScalarPowNode>(left, right);
    }
  }

  void visit(const ScalarExpNode* node) override {
    auto x = map.at(node->x);
    auto x_constant = std::dynamic_pointer_cast<const ScalarConstantNode>(x);

    // {exp(k), k3}, // constant fold
    if (x_constant) {
      rewritten = std::make_shared<ScalarConstantNode>(
          std::exp(x_constant->constant_value));
    }

    // {exp(log(x)), x},
    else if (auto x_log = std::dynamic_pointer_cast<const ScalarLogNode>(x)) {
      rewritten = x_log->x;
    }

    else if (x == node->x) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<ScalarExpNode>(x);
    }
  }

  void visit(const ScalarLogNode* node) override {
    auto x = map.at(node->x);
    auto x_constant = std::dynamic_pointer_cast<const ScalarConstantNode>(x);

    // {log(k), k3}, // constant fold
    if (x_constant) {
      rewritten = std::make_shared<ScalarConstantNode>(
          std::log(x_constant->constant_value));
    }

    // {log(exp(x)), x},
    else if (auto x_exp = std::dynamic_pointer_cast<const ScalarExpNode>(x)) {
      rewritten = x_exp->x;
    }

    else if (x == node->x) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<ScalarLogNode>(x);
    }
  }

  void visit(const ScalarAtanNode* node) override {
    auto x = map.at(node->x);
    auto x_constant = std::dynamic_pointer_cast<const ScalarConstantNode>(x);

    // {atan(k), k3}, // constant fold
    if (x_constant) {
      rewritten = std::make_shared<ScalarConstantNode>(
          std::atan(x_constant->constant_value));
    }

    else if (x == node->x) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<ScalarAtanNode>(x);
    }
  }

  void visit(const ScalarLgammaNode* node) override {
    auto x = map.at(node->x);
    auto x_constant = std::dynamic_pointer_cast<const ScalarConstantNode>(x);

    // {lgamma(k), k3}, // constant fold
    if (x_constant) {
      rewritten = std::make_shared<ScalarConstantNode>(
          std::lgamma(x_constant->constant_value));
    }

    else if (x == node->x) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<ScalarLgammaNode>(x);
    }
  }

  void visit(const ScalarPolygammaNode* node) override {
    auto n = map.at(node->n);
    auto x = map.at(node->x);
    auto n_constant = std::dynamic_pointer_cast<const ScalarConstantNode>(n);
    auto x_constant = std::dynamic_pointer_cast<const ScalarConstantNode>(x);

    // {lgamma(k), k3}, // constant fold
    if (n_constant && x_constant) {
      auto value = boost::math::polygamma(
          n_constant->constant_value, x_constant->constant_value);
      rewritten = std::make_shared<ScalarConstantNode>(value);
    }

    else if (n == node->n && x == node->x) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<ScalarPolygammaNode>(n, x);
    }
  }

  void visit(const ScalarLog1pNode* node) override {
    auto x = map.at(node->x);
    auto x_constant = std::dynamic_pointer_cast<const ScalarConstantNode>(x);

    // {log1p(k), k3}, // constant fold
    if (x_constant) {
      rewritten = std::make_shared<ScalarConstantNode>(
          std::log1p(x_constant->constant_value));
    }

    else if (x == node->x) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<ScalarLog1pNode>(x);
    }
  }

  void visit(const ScalarIfEqualNode* node) override {
    auto a = map.at(node->a);
    auto b = map.at(node->b);
    auto c = map.at(node->c);
    auto d = map.at(node->d);
    auto a_constant = std::dynamic_pointer_cast<const ScalarConstantNode>(a);
    auto b_constant = std::dynamic_pointer_cast<const ScalarConstantNode>(b);

    if (a_constant && b_constant) {
      rewritten =
          (a_constant->constant_value == b_constant->constant_value) ? c : d;
    }

    else if (a == node->a && b == node->b && c == node->c && d == node->d) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<ScalarIfEqualNode>(a, b, c, d);
    }
  }

  void visit(const ScalarIfLessNode* node) override {
    auto a = map.at(node->a);
    auto b = map.at(node->b);
    auto c = map.at(node->c);
    auto d = map.at(node->d);
    auto a_constant = std::dynamic_pointer_cast<const ScalarConstantNode>(a);
    auto b_constant = std::dynamic_pointer_cast<const ScalarConstantNode>(b);

    if (a_constant && b_constant) {
      rewritten =
          (a_constant->constant_value < b_constant->constant_value) ? c : d;
    }

    else if (a == node->a && b == node->b && c == node->c && d == node->d) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<ScalarIfLessNode>(a, b, c, d);
    }
  }

  void visit(const DistributionNormalNode* node) override {
    auto mean = map.at(node->mean);
    auto stddev = map.at(node->stddev);

    if (mean == node->mean && stddev == node->stddev) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<DistributionNormalNode>(mean, stddev);
    }
  }

  void visit(const DistributionHalfNormalNode* node) override {
    auto stddev = map.at(node->stddev);

    if (stddev == node->stddev) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<DistributionHalfNormalNode>(stddev);
    }
  }

  void visit(const DistributionBetaNode* node) override {
    auto a = map.at(node->a);
    auto b = map.at(node->b);

    if (a == node->a && b == node->b) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<DistributionBetaNode>(a, b);
    }
  }

  void visit(const DistributionBernoulliNode* node) override {
    auto prob = map.at(node->prob);

    if (prob == node->prob) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<DistributionBernoulliNode>(prob);
    }
  }

  void visit(const DistributionExponentialNode* node) override {
    auto rate = map.at(node->rate);

    if (rate == node->rate) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<DistributionExponentialNode>(rate);
    }
  }
};

} // namespace

namespace beanmachine::minibmg {

Nodep rewrite_node(const Nodep& node, NodeNodeValueMap& map);

std::unordered_map<Nodep, Nodep> opt_map(std::vector<Nodep> roots) {
  std::vector<Nodep> sorted;
  if (!topological_sort<Nodep>(
          {roots.begin(), roots.end()}, in_nodes, sorted)) {
    throw std::invalid_argument("graph has a cycle");
  }
  std::reverse(sorted.begin(), sorted.end());

  // a value-based, map, which treats semantically identical nodes as the same.
  NodeNodeValueMap map;
  RewriteOneVisitor v{map};

  for (auto& node : sorted) {
    v.rewrite_node(node);
  }

  // We also build a map that uses object (pointer) identity to find elements,
  // so that clients are not using recursive node equality tests.
  std::unordered_map<Nodep, Nodep> identity_map;
  for (auto& node : sorted) {
    identity_map.insert({node, map.at(node)});
  }

  return identity_map;
}

} // namespace beanmachine::minibmg
