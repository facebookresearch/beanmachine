/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/dedup.h"
#include <map>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include "beanmachine/minibmg/node.h"
#include "beanmachine/minibmg/topological.h"

namespace {

using namespace beanmachine::minibmg;

// A visitor whose job is to perform one step in deduplication, by deduplicating
// a single given node assuming all nodes reachable from its inputs have been
// deduplicated, and those deduplications recorded in the given map.
//
// Deduplication is not done recursively as we want to bound the depth of
// recursive execution at runtime.
class NodeReplacementVisitor : NodeVisitor {
 private:
  NodeNodeValueMap& map;
  Nodep original;
  Nodep result;

 public:
  explicit NodeReplacementVisitor(NodeNodeValueMap& map) : map{map} {}

  Nodep rewrite(Nodep node) {
    // We save the original so that, in case no rewriting is needed, we can
    // return it as the new `result`.  We cannot return "node" in the visit
    // method in that case, as we are trying to return smart pointers rather
    // than raw pointers.
    original = node;
    node->accept(*this);
    return result;
  }

 private:
  void visit(const ScalarConstantNode*) override {
    result = original;
  }
  void visit(const ScalarVariableNode*) override {
    result = original;
  }
  void visit(const ScalarSampleNode* node) override {
    const DistributionNodep dist = map.at(node->distribution);
    if (dist == node->distribution) {
      result = original;
    } else {
      result = std::make_shared<ScalarSampleNode>(dist, node->rvid);
    }
  }
  void visit(const ScalarAddNode* node) override {
    const ScalarNodep left = map.at(node->left);
    const ScalarNodep right = map.at(node->right);
    if (left == node->left && right == node->right) {
      result = original;
    } else {
      result = std::make_shared<ScalarAddNode>(left, right);
    }
  }
  void visit(const ScalarSubtractNode* node) override {
    const ScalarNodep left = map.at(node->left);
    const ScalarNodep right = map.at(node->right);
    if (left == node->left && right == node->right) {
      result = original;
    } else {
      result = std::make_shared<ScalarSubtractNode>(left, right);
    }
  }
  void visit(const ScalarNegateNode* node) override {
    const ScalarNodep x = map.at(node->x);
    if (x == node->x) {
      result = original;
    } else {
      result = std::make_shared<ScalarNegateNode>(x);
    }
  }
  void visit(const ScalarMultiplyNode* node) override {
    const ScalarNodep left = map.at(node->left);
    const ScalarNodep right = map.at(node->right);
    if (left == node->left && right == node->right) {
      result = original;
    } else {
      result = std::make_shared<ScalarMultiplyNode>(left, right);
    }
  }
  void visit(const ScalarDivideNode* node) override {
    const ScalarNodep left = map.at(node->left);
    const ScalarNodep right = map.at(node->right);
    if (left == node->left && right == node->right) {
      result = original;
    } else {
      result = std::make_shared<ScalarDivideNode>(left, right);
    }
  }
  void visit(const ScalarPowNode* node) override {
    const ScalarNodep left = map.at(node->left);
    const ScalarNodep right = map.at(node->right);
    if (left == node->left && right == node->right) {
      result = original;
    } else {
      result = std::make_shared<ScalarPowNode>(left, right);
    }
  }
  void visit(const ScalarExpNode* node) override {
    const ScalarNodep x = map.at(node->x);
    if (x == node->x) {
      result = original;
    } else {
      result = std::make_shared<ScalarExpNode>(x);
    }
  }
  void visit(const ScalarLogNode* node) override {
    const ScalarNodep x = map.at(node->x);
    if (x == node->x) {
      result = original;
    } else {
      result = std::make_shared<ScalarLogNode>(x);
    }
  }
  void visit(const ScalarAtanNode* node) override {
    const ScalarNodep x = map.at(node->x);
    if (x == node->x) {
      result = original;
    } else {
      result = std::make_shared<ScalarAtanNode>(x);
    }
  }
  void visit(const ScalarLgammaNode* node) override {
    const ScalarNodep x = map.at(node->x);
    if (x == node->x) {
      result = original;
    } else {
      result = std::make_shared<ScalarLgammaNode>(x);
    }
  }
  void visit(const ScalarPolygammaNode* node) override {
    const ScalarNodep n = map.at(node->n);
    const ScalarNodep x = map.at(node->x);
    if (n == node->n && x == node->x) {
      result = original;
    } else {
      result = std::make_shared<ScalarPolygammaNode>(n, x);
    }
  }
  void visit(const ScalarLog1pNode* node) override {
    const ScalarNodep x = map.at(node->x);
    if (x == node->x) {
      result = original;
    } else {
      result = std::make_shared<ScalarLog1pNode>(x);
    }
  }
  void visit(const ScalarIfEqualNode* node) override {
    const ScalarNodep a = map.at(node->a);
    const ScalarNodep b = map.at(node->b);
    const ScalarNodep c = map.at(node->c);
    const ScalarNodep d = map.at(node->d);
    if (a == node->a && b == node->b && c == node->c && d == node->d) {
      result = original;
    } else {
      result = std::make_shared<ScalarIfEqualNode>(a, b, c, d);
    }
  }
  void visit(const ScalarIfLessNode* node) override {
    const ScalarNodep a = map.at(node->a);
    const ScalarNodep b = map.at(node->b);
    const ScalarNodep c = map.at(node->c);
    const ScalarNodep d = map.at(node->d);
    if (a == node->a && b == node->b && c == node->c && d == node->d) {
      result = original;
    } else {
      result = std::make_shared<ScalarIfLessNode>(a, b, c, d);
    }
  }
  void visit(const DistributionNormalNode* node) override {
    const ScalarNodep mean = map.at(node->mean);
    const ScalarNodep stddev = map.at(node->stddev);
    if (mean == node->mean && stddev == node->stddev) {
      result = original;
    } else {
      result = std::make_shared<DistributionNormalNode>(mean, stddev);
    }
  }
  void visit(const DistributionHalfNormalNode* node) override {
    const ScalarNodep stddev = map.at(node->stddev);
    if (stddev == node->stddev) {
      result = original;
    } else {
      result = std::make_shared<DistributionHalfNormalNode>(stddev);
    }
  }
  void visit(const DistributionBetaNode* node) override {
    const ScalarNodep a = map.at(node->a);
    const ScalarNodep b = map.at(node->b);
    if (a == node->a && b == node->b) {
      result = original;
    } else {
      result = std::make_shared<DistributionBetaNode>(a, b);
    }
  }
  void visit(const DistributionBernoulliNode* node) override {
    const ScalarNodep prob = map.at(node->prob);
    if (prob == node->prob) {
      result = original;
    } else {
      result = std::make_shared<DistributionBernoulliNode>(prob);
    }
  }
  void visit(const DistributionExponentialNode* node) override {
    const ScalarNodep rate = map.at(node->rate);
    if (rate == node->rate) {
      result = original;
    } else {
      result = std::make_shared<DistributionExponentialNode>(rate);
    }
  }
};

} // namespace

namespace beanmachine::minibmg {

// Take a set of root nodes as input, and return a map of deduplicated nodes,
// which maps from a node in the transitive closure of the input to a
// corresponding node in the transitive closure of the deduplicated graph.
std::unordered_map<Nodep, Nodep> dedup_map(std::vector<Nodep> roots) {
  // a value-based, map, which treats semantically identical nodes as the same.
  NodeNodeValueMap map;

  // We also build a map that uses object (pointer) identity to find elements,
  // so that operations in clients are not using recursive equality operations.
  std::unordered_map<Nodep, Nodep> identity_map;

  std::vector<Nodep> sorted;
  if (!topological_sort<Nodep>(
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
