/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/dedup2.h"
#include <map>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include "beanmachine/minibmg/node2.h"
#include "beanmachine/minibmg/topological.h"

namespace {

using namespace beanmachine::minibmg;

// A visitor whose job is to perform one step in deduplication, by deduplicating
// a single given node assuming all nodes reachable from its inputs have been
// deduplicated, and those deduplications recorded in the given map.
//
// Deduplication is not done recursively as we want to bound the depth of
// recursive execution at runtime.
class NodeReplacementVisitor : Node2Visitor {
 private:
  Node2Node2ValueMap& map;
  Node2p original;
  Node2p result;

 public:
  explicit NodeReplacementVisitor(Node2Node2ValueMap& map) : map{map} {}

  Node2p rewrite(Node2p node) {
    // We save the original so that, in case no rewriting is needed, we can
    // return it as the new `result`.  We cannot return "node" in the visit
    // method in that case, as we are trying to return smart pointers rather
    // than raw pointers.
    original = node;
    node->accept(*this);
    return result;
  }

 private:
  void visit(const ScalarConstantNode2*) override {
    result = original;
  }
  void visit(const ScalarVariableNode2*) override {
    result = original;
  }
  void visit(const ScalarSampleNode2* node) override {
    const DistributionNode2p dist = map.at(node->distribution);
    if (dist == node->distribution) {
      result = original;
    } else {
      result = std::make_shared<ScalarSampleNode2>(dist);
    }
  }
  void visit(const ScalarAddNode2* node) override {
    const ScalarNode2p left = map.at(node->left);
    const ScalarNode2p right = map.at(node->right);
    if (left == node->left && right == node->right) {
      result = original;
    } else {
      result = std::make_shared<ScalarAddNode2>(left, right);
    }
  }
  void visit(const ScalarSubtractNode2* node) override {
    const ScalarNode2p left = map.at(node->left);
    const ScalarNode2p right = map.at(node->right);
    if (left == node->left && right == node->right) {
      result = original;
    } else {
      result = std::make_shared<ScalarSubtractNode2>(left, right);
    }
  }
  void visit(const ScalarNegateNode2* node) override {
    const ScalarNode2p x = map.at(node->x);
    if (x == node->x) {
      result = original;
    } else {
      result = std::make_shared<ScalarNegateNode2>(x);
    }
  }
  void visit(const ScalarMultiplyNode2* node) override {
    const ScalarNode2p left = map.at(node->left);
    const ScalarNode2p right = map.at(node->right);
    if (left == node->left && right == node->right) {
      result = original;
    } else {
      result = std::make_shared<ScalarMultiplyNode2>(left, right);
    }
  }
  void visit(const ScalarDivideNode2* node) override {
    const ScalarNode2p left = map.at(node->left);
    const ScalarNode2p right = map.at(node->right);
    if (left == node->left && right == node->right) {
      result = original;
    } else {
      result = std::make_shared<ScalarDivideNode2>(left, right);
    }
  }
  void visit(const ScalarPowNode2* node) override {
    const ScalarNode2p left = map.at(node->left);
    const ScalarNode2p right = map.at(node->right);
    if (left == node->left && right == node->right) {
      result = original;
    } else {
      result = std::make_shared<ScalarPowNode2>(left, right);
    }
  }
  void visit(const ScalarExpNode2* node) override {
    const ScalarNode2p x = map.at(node->x);
    if (x == node->x) {
      result = original;
    } else {
      result = std::make_shared<ScalarExpNode2>(x);
    }
  }
  void visit(const ScalarLogNode2* node) override {
    const ScalarNode2p x = map.at(node->x);
    if (x == node->x) {
      result = original;
    } else {
      result = std::make_shared<ScalarLogNode2>(x);
    }
  }
  void visit(const ScalarAtanNode2* node) override {
    const ScalarNode2p x = map.at(node->x);
    if (x == node->x) {
      result = original;
    } else {
      result = std::make_shared<ScalarAtanNode2>(x);
    }
  }
  void visit(const ScalarLgammaNode2* node) override {
    const ScalarNode2p x = map.at(node->x);
    if (x == node->x) {
      result = original;
    } else {
      result = std::make_shared<ScalarLgammaNode2>(x);
    }
  }
  void visit(const ScalarPolygammaNode2* node) override {
    const ScalarNode2p n = map.at(node->n);
    const ScalarNode2p x = map.at(node->x);
    if (n == node->n && x == node->x) {
      result = original;
    } else {
      result = std::make_shared<ScalarPolygammaNode2>(n, x);
    }
  }
  void visit(const ScalarIfEqualNode2* node) override {
    const ScalarNode2p a = map.at(node->a);
    const ScalarNode2p b = map.at(node->b);
    const ScalarNode2p c = map.at(node->c);
    const ScalarNode2p d = map.at(node->d);
    if (a == node->a && b == node->b && c == node->c && d == node->d) {
      result = original;
    } else {
      result = std::make_shared<ScalarIfEqualNode2>(a, b, c, d);
    }
  }
  void visit(const ScalarIfLessNode2* node) override {
    const ScalarNode2p a = map.at(node->a);
    const ScalarNode2p b = map.at(node->b);
    const ScalarNode2p c = map.at(node->c);
    const ScalarNode2p d = map.at(node->d);
    if (a == node->a && b == node->b && c == node->c && d == node->d) {
      result = original;
    } else {
      result = std::make_shared<ScalarIfLessNode2>(a, b, c, d);
    }
  }
  void visit(const DistributionNormalNode2* node) override {
    const ScalarNode2p mean = map.at(node->mean);
    const ScalarNode2p stddev = map.at(node->stddev);
    if (mean == node->mean && stddev == node->stddev) {
      result = original;
    } else {
      result = std::make_shared<DistributionNormalNode2>(mean, stddev);
    }
  }
  void visit(const DistributionHalfNormalNode2* node) override {
    const ScalarNode2p stddev = map.at(node->stddev);
    if (stddev == node->stddev) {
      result = original;
    } else {
      result = std::make_shared<DistributionHalfNormalNode2>(stddev);
    }
  }
  void visit(const DistributionBetaNode2* node) override {
    const ScalarNode2p a = map.at(node->a);
    const ScalarNode2p b = map.at(node->b);
    if (a == node->a && b == node->b) {
      result = original;
    } else {
      result = std::make_shared<DistributionBetaNode2>(a, b);
    }
  }
  void visit(const DistributionBernoulliNode2* node) override {
    const ScalarNode2p prob = map.at(node->prob);
    if (prob == node->prob) {
      result = original;
    } else {
      result = std::make_shared<DistributionBernoulliNode2>(prob);
    }
  }
};

} // namespace

namespace beanmachine::minibmg {

// Take a set of root nodes as input, and return a map of deduplicated nodes,
// which maps from a node in the transitive closure of the input to a
// corresponding node in the transitive closure of the deduplicated graph.
std::unordered_map<Node2p, Node2p> dedup_map(std::vector<Node2p> roots) {
  // a value-based, map, which treats semantically identical nodes as the same.
  Node2Node2ValueMap map;

  // We also build a map that uses object (pointer) identity to find elements,
  // so that operations in clients are not using recursive equality operations.
  std::unordered_map<Node2p, Node2p> identity_map;

  std::vector<Node2p> sorted;
  if (!topological_sort<Node2p>(
          {roots.begin(), roots.end()}, in_nodes, sorted)) {
    throw std::invalid_argument("graph has a cycle");
  }
  std::reverse(sorted.begin(), sorted.end());

  // A node replacement rewriter that offers a replacement for a node by
  // rewriting its immediate inputs if necessary.  Note that it keep a reference
  // to the map.
  NodeReplacementVisitor node_rewriter{map};

  // Compute a replacement for each node.
  for (auto& node : sorted) {
    auto found = map.find(node);
    if (found != map.end()) {
      auto mapping = found->second;
      identity_map.insert({node, mapping});
    } else {
      auto rewritten = node_rewriter.rewrite(node);
      map.insert(node, rewritten);
      identity_map.insert({node, rewritten});
    }
  }

  return identity_map;
}

} // namespace beanmachine::minibmg
