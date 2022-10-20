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
#include "beanmachine/minibmg/node2.h"
#include "beanmachine/minibmg/topological.h"

namespace {

using namespace beanmachine::minibmg;

Node2pIdentityEquals same{};

// This is a temporary hack to perform some local optimizations on the a graph
// node. Ultimately, these should be organized into a rewriter based on tree
// automata, which will make the rewriters much easier to maintain and much
// faster.  For now we hand-implement a few rules by brute force.  The one-line
// comment before each transformation shows what the rule would look like in a
// hypothetical rewriting system.
class RewriteOneVisitor : Node2Visitor {
 private:
  Node2Node2ValueMap& map;
  Node2p original;
  Node2p rewritten;

 public:
  explicit RewriteOneVisitor(Node2Node2ValueMap& map) : map{map} {}
  Node2p rewrite_one(const Node2p& node) {
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

  ScalarNode2p rewrite_scalar_node(const ScalarNode2p& node) {
    return std::dynamic_pointer_cast<const ScalarNode2>(rewrite_node(node));
  }

  // The following method may be useful in debugging the problematic case
  // that the rewrite_one method returns nested nodes that it does not
  // place in the map. It is commented out in normal use, but when the
  // optimizer throws an exception because a node is not in the map, this
  // will likely be helpful in finding the problem.
  bool check_children(const Node2p& node) {
    for (auto in : in_nodes(node)) {
      if (!map.contains(in) || map.at(in) == nullptr) {
        return false;
      }
    }
    return true;
  }

  // Call the rewriter repeatedly on a node until a fixed-point is reached, and
  // then place the result in the node-value-based map.
  Node2p rewrite_node(const Node2p& node) {
    // check_children(node, map);
    Node2p rewritten = node;
    while (true) {
      const Node2p n = rewrite_one(rewritten);
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
  void visit(const ScalarConstantNode2*) override {
    rewritten = original;
  }

  void visit(const ScalarVariableNode2*) override {
    rewritten = original;
  }

  void visit(const ScalarSampleNode2* node) override {
    auto d = map.at(node->distribution);
    if (node->distribution != d) {
      rewritten = std::make_shared<ScalarSampleNode2>(d, node->rvid);
    }

    else {
      rewritten = original;
    }
  }

  void visit(const ScalarAddNode2* node) override {
    auto left = map.at(node->left);
    auto right = map.at(node->right);
    auto left_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode2>(left);
    auto right_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode2>(right);

    // {k1 + k2, k3}, // constant fold
    if (left_constant && right_constant) {
      rewritten = std::make_shared<ScalarConstantNode2>(
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
      ScalarNode2p two =
          rewrite_scalar_node(std::make_shared<ScalarConstantNode2>(2));
      rewritten = std::make_shared<ScalarMultiplyNode2>(two, left);
    }

    else if (left != node->left || right != node->right) {
      rewritten = std::make_shared<ScalarAddNode2>(left, right);
    }

    else {
      rewritten = original;
    }
  }

  void visit(const ScalarSubtractNode2* node) override {
    auto left = map.at(node->left);
    auto right = map.at(node->right);
    auto left_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode2>(left);
    auto right_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode2>(right);

    // {k1 - k2, k3}, // constant fold
    if (left_constant && right_constant) {
      rewritten = std::make_shared<ScalarConstantNode2>(
          left_constant->constant_value - right_constant->constant_value);
    }

    // {0 - x, -x},
    else if (left_constant && left_constant->constant_value == 0) {
      rewritten = std::make_shared<ScalarNegateNode2>(right);
    }

    // {x - 0, x},
    else if (right_constant && right_constant->constant_value == 0) {
      rewritten = left;
    }

    // {x - x, 0},
    else if (same(left, right)) {
      rewritten = std::make_shared<ScalarConstantNode2>(0);
    }

    else if (left != node->left || right != node->right) {
      rewritten = std::make_shared<ScalarSubtractNode2>(left, right);
    }

    else {
      rewritten = original;
    }
  }

  void visit(const ScalarNegateNode2* node) override {
    auto x = map.at(node->x);
    auto x_constant = std::dynamic_pointer_cast<const ScalarConstantNode2>(x);

    // {-k, k3}, // constant fold
    if (x_constant) {
      rewritten =
          std::make_shared<ScalarConstantNode2>(-x_constant->constant_value);
    }

    // {--x, x},
    else if (
        auto x_negate = std::dynamic_pointer_cast<const ScalarNegateNode2>(x)) {
      rewritten = x_negate->x;
    }

    else if (x != node->x) {
      rewritten = std::make_shared<ScalarNegateNode2>(x);
    }

    else {
      rewritten = original;
    }
  }

  void visit(const ScalarMultiplyNode2* node) override {
    auto left = map.at(node->left);
    auto right = map.at(node->right);
    auto left_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode2>(left);
    auto right_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode2>(right);

    // {k1 * k2, k3}, // constant fold
    if (left_constant && right_constant) {
      rewritten = std::make_shared<ScalarConstantNode2>(
          left_constant->constant_value * right_constant->constant_value);
    }

    // {0 * x, 0},
    else if (left_constant && left_constant->constant_value == 0) {
      rewritten = left;
    }

    // {-1 * x, -x},
    else if (left_constant && left_constant->constant_value == -1) {
      rewritten = std::make_shared<ScalarNegateNode2>(right);
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
                 std::dynamic_pointer_cast<const ScalarMultiplyNode2>(right);
             left_constant && right_multiply) {
      if (auto right_left_constant =
              std::dynamic_pointer_cast<const ScalarConstantNode2>(
                  right_multiply->left);
          right_left_constant) {
        auto k3 = rewrite_scalar_node(std::make_shared<ScalarConstantNode2>(
            left_constant->constant_value *
            right_left_constant->constant_value));
        rewritten =
            std::make_shared<ScalarMultiplyNode2>(k3, right_multiply->right);
      }
    }

    if (rewritten == nullptr) {
      if (left != node->left || right != node->right) {
        rewritten = std::make_shared<ScalarMultiplyNode2>(left, right);
      }

      else {
        rewritten = original;
      }
    }
  }

  void visit(const ScalarDivideNode2* node) override {
    auto left = map.at(node->left);
    auto right = map.at(node->right);
    auto left_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode2>(left);
    auto right_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode2>(right);

    // {k1 / k2, k3}, // constant fold
    if (left_constant && right_constant) {
      rewritten = std::make_shared<ScalarConstantNode2>(
          left_constant->constant_value / right_constant->constant_value);
    }

    // {0 / x, 0},
    else if (left_constant && left_constant->constant_value == 0) {
      rewritten = left;
    }

    // {x / k, (1/k) * x},
    else if (right_constant) {
      auto k3 = rewrite_scalar_node(std::make_shared<ScalarConstantNode2>(
          1 / right_constant->constant_value));
      rewritten = std::make_shared<ScalarMultiplyNode2>(k3, left);
    }

    else if (left != node->left || right != node->right) {
      rewritten = std::make_shared<ScalarDivideNode2>(left, right);
    }

    else {
      rewritten = original;
    }
  }

  void visit(const ScalarPowNode2* node) override {
    auto left = map.at(node->left);
    auto right = map.at(node->right);
    auto left_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode2>(left);
    auto right_constant =
        std::dynamic_pointer_cast<const ScalarConstantNode2>(right);

    // {pow(k1, k2), k3}, // constant fold
    if (left_constant && right_constant) {
      auto k3 = std::pow(
          left_constant->constant_value, right_constant->constant_value);
      rewritten = std::make_shared<ScalarConstantNode2>(k3);
    }

    // {pow(x, 1), x},
    else if (right_constant && right_constant->constant_value == 1) {
      rewritten = left;
    }

    else if (left == node->left && right == node->right) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<ScalarPowNode2>(left, right);
    }
  }

  void visit(const ScalarExpNode2* node) override {
    auto x = map.at(node->x);
    auto x_constant = std::dynamic_pointer_cast<const ScalarConstantNode2>(x);

    // {exp(k), k3}, // constant fold
    if (x_constant) {
      rewritten = std::make_shared<ScalarConstantNode2>(
          std::exp(x_constant->constant_value));
    }

    // {exp(log(x)), x},
    else if (auto x_log = std::dynamic_pointer_cast<const ScalarLogNode2>(x)) {
      rewritten = x_log->x;
    }

    else if (x == node->x) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<ScalarExpNode2>(x);
    }
  }

  void visit(const ScalarLogNode2* node) override {
    auto x = map.at(node->x);
    auto x_constant = std::dynamic_pointer_cast<const ScalarConstantNode2>(x);

    // {log(k), k3}, // constant fold
    if (x_constant) {
      rewritten = std::make_shared<ScalarConstantNode2>(
          std::log(x_constant->constant_value));
    }

    // {log(exp(x)), x},
    else if (auto x_exp = std::dynamic_pointer_cast<const ScalarExpNode2>(x)) {
      rewritten = x_exp->x;
    }

    else if (x == node->x) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<ScalarLogNode2>(x);
    }
  }

  void visit(const ScalarAtanNode2* node) override {
    auto x = map.at(node->x);
    auto x_constant = std::dynamic_pointer_cast<const ScalarConstantNode2>(x);

    // {atan(k), k3}, // constant fold
    if (x_constant) {
      rewritten = std::make_shared<ScalarConstantNode2>(
          std::atan(x_constant->constant_value));
    }

    else if (x == node->x) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<ScalarAtanNode2>(x);
    }
  }

  void visit(const ScalarLgammaNode2* node) override {
    auto x = map.at(node->x);
    auto x_constant = std::dynamic_pointer_cast<const ScalarConstantNode2>(x);

    // {lgamma(k), k3}, // constant fold
    if (x_constant) {
      rewritten = std::make_shared<ScalarConstantNode2>(
          std::lgamma(x_constant->constant_value));
    }

    else if (x == node->x) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<ScalarLgammaNode2>(x);
    }
  }

  void visit(const ScalarPolygammaNode2* node) override {
    auto n = map.at(node->n);
    auto x = map.at(node->x);
    auto n_constant = std::dynamic_pointer_cast<const ScalarConstantNode2>(n);
    auto x_constant = std::dynamic_pointer_cast<const ScalarConstantNode2>(x);

    // {lgamma(k), k3}, // constant fold
    if (n_constant && x_constant) {
      auto value = boost::math::polygamma(
          n_constant->constant_value, x_constant->constant_value);
      rewritten = std::make_shared<ScalarConstantNode2>(value);
    }

    else if (n == node->n && x == node->x) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<ScalarPolygammaNode2>(n, x);
    }
  }

  void visit(const ScalarIfEqualNode2* node) override {
    auto a = map.at(node->a);
    auto b = map.at(node->b);
    auto c = map.at(node->c);
    auto d = map.at(node->d);
    auto a_constant = std::dynamic_pointer_cast<const ScalarConstantNode2>(a);
    auto b_constant = std::dynamic_pointer_cast<const ScalarConstantNode2>(b);

    if (a_constant && b_constant) {
      rewritten =
          (a_constant->constant_value == b_constant->constant_value) ? c : d;
    }

    else if (a == node->a && b == node->b && c == node->c && d == node->d) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<ScalarIfEqualNode2>(a, b, c, d);
    }
  }

  void visit(const ScalarIfLessNode2* node) override {
    auto a = map.at(node->a);
    auto b = map.at(node->b);
    auto c = map.at(node->c);
    auto d = map.at(node->d);
    auto a_constant = std::dynamic_pointer_cast<const ScalarConstantNode2>(a);
    auto b_constant = std::dynamic_pointer_cast<const ScalarConstantNode2>(b);

    if (a_constant && b_constant) {
      rewritten =
          (a_constant->constant_value < b_constant->constant_value) ? c : d;
    }

    else if (a == node->a && b == node->b && c == node->c && d == node->d) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<ScalarIfLessNode2>(a, b, c, d);
    }
  }

  void visit(const DistributionNormalNode2* node) override {
    auto mean = map.at(node->mean);
    auto stddev = map.at(node->stddev);

    if (mean == node->mean && stddev == node->stddev) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<DistributionNormalNode2>(mean, stddev);
    }
  }

  void visit(const DistributionHalfNormalNode2* node) override {
    auto stddev = map.at(node->stddev);

    if (stddev == node->stddev) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<DistributionHalfNormalNode2>(stddev);
    }
  }

  void visit(const DistributionBetaNode2* node) override {
    auto a = map.at(node->a);
    auto b = map.at(node->b);

    if (a == node->a && b == node->b) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<DistributionBetaNode2>(a, b);
    }
  }

  void visit(const DistributionBernoulliNode2* node) override {
    auto prob = map.at(node->prob);

    if (prob == node->prob) {
      rewritten = original;
    }

    else {
      rewritten = std::make_shared<DistributionBernoulliNode2>(prob);
    }
  }
};

} // namespace

namespace beanmachine::minibmg {

Node2p rewrite_node(const Node2p& node, Node2Node2ValueMap& map);

std::unordered_map<Node2p, Node2p> opt_map(std::vector<Node2p> roots) {
  std::vector<Node2p> sorted;
  if (!topological_sort<Node2p>(
          {roots.begin(), roots.end()}, in_nodes, sorted)) {
    throw std::invalid_argument("graph has a cycle");
  }
  std::reverse(sorted.begin(), sorted.end());

  // a value-based, map, which treats semantically identical nodes as the same.
  Node2Node2ValueMap map;
  RewriteOneVisitor v{map};

  for (auto& node : sorted) {
    v.rewrite_node(node);
  }

  // We also build a map that uses object (pointer) identity to find elements,
  // so that clients are not using recursive node equality tests.
  std::unordered_map<Node2p, Node2p> identity_map;
  for (auto& node : sorted) {
    identity_map.insert({node, map.at(node)});
  }

  return identity_map;
}

} // namespace beanmachine::minibmg
