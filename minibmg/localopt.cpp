/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/localopt.h"
#include <memory>
#include <unordered_map>
#include "beanmachine/minibmg/node.h"
#include "beanmachine/minibmg/operator.h"
#include "beanmachine/minibmg/topological.h"

namespace {

using namespace beanmachine::minibmg;
double k(Nodep node) {
  return std::dynamic_pointer_cast<const ConstantNode>(node)->value;
}

NodepIdentityEquals same{};

Nodep make_operator(Operator op, std::vector<Nodep> in_nodes) {
  return std::make_shared<OperatorNode>(in_nodes, op, Type::REAL);
}

} // namespace

namespace beanmachine::minibmg {

Nodep rewrite_node(const Nodep& node, NodeValueMap<Nodep>& map);

// This is a temporary hack to perform some local optimizations on the a graph
// node. Ultimately, these should be organized into a rewriter based on tree
// automata, which will make the rewriters much easier to maintain and much
// faster.  For now we hand-implement a few rules by brute force.  The one-line
// comment before each transformation shows what the rule would look like in a
// hypothetical rewriting system.
Nodep rewrite_one(const Nodep& node, NodeValueMap<Nodep>& map) {
  // A semantically equivalent node was already rewritten.
  if (auto found = map.find(node); found != map.end()) {
    return found->second;
  }
  switch (node->op) {
    case Operator::ADD: {
      auto op = std::dynamic_pointer_cast<const OperatorNode>(node);
      // {k1 + k2, k3}, // constant fold
      auto left = map.at(op->in_nodes[0]);
      auto right = map.at(op->in_nodes[1]);
      if (left->op == Operator::CONSTANT && right->op == Operator::CONSTANT) {
        return std::make_shared<ConstantNode>(k(left) + k(right));
      }
      // {0 + x, x},
      if (left->op == Operator::CONSTANT && k(left) == 0) {
        return right;
      }
      // {x + 0, x},
      if (right->op == Operator::CONSTANT && k(right) == 0) {
        return left;
      }
      // {x + x, 2 * x},
      if (same(left, right)) {
        auto two = rewrite_node(std::make_shared<ConstantNode>(2), map);
        return make_operator(Operator::MULTIPLY, {two, left});
      }
      if (left == op->in_nodes[0] && right == op->in_nodes[1]) {
        return node;
      }
      return make_operator(Operator::ADD, {left, right});
    }

    case Operator::SUBTRACT: {
      auto op = std::dynamic_pointer_cast<const OperatorNode>(node);
      auto left = map.at(op->in_nodes[0]);
      auto right = map.at(op->in_nodes[1]);
      // {k1 - k2, k3}, // constant fold
      if (left->op == Operator::CONSTANT && right->op == Operator::CONSTANT) {
        return std::make_shared<ConstantNode>(k(left) - k(right));
      }
      // {0 - x, -x},
      if (left->op == Operator::CONSTANT && k(left) == 0) {
        return make_operator(Operator::NEGATE, {right});
      }
      // {x - 0, x},
      if (right->op == Operator::CONSTANT && k(right) == 0) {
        return left;
      }
      // {x - x, 0},
      if (same(left, right)) {
        return std::make_shared<ConstantNode>(0);
      }
      if (left == op->in_nodes[0] && right == op->in_nodes[1]) {
        return node;
      }
      return make_operator(Operator::SUBTRACT, {left, right});
    }

    case Operator::NEGATE: {
      auto op = std::dynamic_pointer_cast<const OperatorNode>(node);
      auto x = map.at(op->in_nodes[0]);
      // {-k, k3}, // constant fold
      if (x->op == Operator::CONSTANT) {
        return std::make_shared<ConstantNode>(-k(x));
      }
      // {--x, x},
      if (x->op == Operator::NEGATE) {
        return std::dynamic_pointer_cast<const OperatorNode>(x)->in_nodes[0];
      }
      if (x == op->in_nodes[0]) {
        return node;
      }
      return make_operator(Operator::NEGATE, {x});
    }

    case Operator::MULTIPLY: {
      auto op = std::dynamic_pointer_cast<const OperatorNode>(node);
      auto left = map.at(op->in_nodes[0]);
      auto right = map.at(op->in_nodes[1]);
      // {k1 * k2, k3}, // constant fold
      if (left->op == Operator::CONSTANT && right->op == Operator::CONSTANT) {
        return std::make_shared<ConstantNode>(k(left) * k(right));
      }
      // {0 * x, 0},
      if (left->op == Operator::CONSTANT && k(left) == 0) {
        return left;
      }
      // {-1 * x, -x},
      if (left->op == Operator::CONSTANT && k(left) == -1) {
        return make_operator(Operator::NEGATE, {right});
      }
      // {1 * x, x},
      if (left->op == Operator::CONSTANT && k(left) == 1) {
        return right;
      }
      // {x * 0, 0},
      if (right->op == Operator::CONSTANT && k(right) == 0) {
        return right;
      }
      // {x * 1, x},
      if (right->op == Operator::CONSTANT && k(right) == 1) {
        return left;
      }
      // {k1 * (k2 * x), k3 * x },
      if (left->op == Operator::CONSTANT && right->op == Operator::MULTIPLY) {
        auto rop = std::dynamic_pointer_cast<const OperatorNode>(node);
        if (rop->in_nodes[0]->op == Operator::CONSTANT) {
          auto k3 = rewrite_node(
              std::make_shared<ConstantNode>(k(left) * k(rop->in_nodes[0])),
              map);
          return make_operator(Operator::MULTIPLY, {k3, rop->in_nodes[1]});
        }
      }
      if (left == op->in_nodes[0] && right == op->in_nodes[1]) {
        return node;
      }
      return make_operator(Operator::MULTIPLY, {left, right});
    }

    case Operator::DIVIDE: {
      auto op = std::dynamic_pointer_cast<const OperatorNode>(node);
      auto left = map.at(op->in_nodes[0]);
      auto right = map.at(op->in_nodes[1]);
      // {k1 / k2, k3}, // constant fold
      if (left->op == Operator::CONSTANT && right->op == Operator::CONSTANT) {
        return std::make_shared<ConstantNode>(k(left) / k(right));
      }
      // {x / k, (1/k) * x},
      if (right->op == Operator::CONSTANT) {
        auto k3 =
            rewrite_node(std::make_shared<ConstantNode>(1 / k(right)), map);
        return make_operator(Operator::MULTIPLY, {k3, left});
      }
      if (left == op->in_nodes[0] && right == op->in_nodes[1]) {
        return node;
      }
      return make_operator(Operator::DIVIDE, {left, right});
    }

    case Operator::POW: {
      auto op = std::dynamic_pointer_cast<const OperatorNode>(node);
      auto left = map.at(op->in_nodes[0]);
      auto right = map.at(op->in_nodes[1]);
      // {pow(k1, k2), k3}, // constant fold
      if (left->op == Operator::CONSTANT && right->op == Operator::CONSTANT) {
        auto k3 = std::pow(k(left), k(right));
        return std::make_shared<ConstantNode>(k3);
      }
      // {pow(x, 1), x},
      if (right->op == Operator::CONSTANT && k(right) == 1) {
        return left;
      }
      if (left == op->in_nodes[0] && right == op->in_nodes[1]) {
        return node;
      }
      return make_operator(Operator::POW, {left, right});
    }

    case Operator::EXP: {
      auto op = std::dynamic_pointer_cast<const OperatorNode>(node);
      auto x = map.at(op->in_nodes[0]);
      // {exp(k), k3}, // constant fold
      if (x->op == Operator::CONSTANT) {
        return std::make_shared<ConstantNode>(std::exp(k(x)));
      }
      // {exp(log(x)), x},
      if (x->op == Operator::LOG) {
        return std::dynamic_pointer_cast<const OperatorNode>(x)->in_nodes[0];
      }
      if (x == op->in_nodes[0]) {
        return node;
      }
      return make_operator(Operator::EXP, {x});
    }

    case Operator::LOG: {
      auto op = std::dynamic_pointer_cast<const OperatorNode>(node);
      auto x = map.at(op->in_nodes[0]);
      // {log(k), k3}, // constant fold
      if (x->op == Operator::CONSTANT) {
        return std::make_shared<ConstantNode>(std::log(k(x)));
      }
      // {log(exp(x)), x},
      if (x->op == Operator::EXP) {
        return std::dynamic_pointer_cast<const OperatorNode>(x)->in_nodes[0];
      }
      if (x == op->in_nodes[0]) {
        return node;
      }
      return make_operator(Operator::LOG, {x});
    }

    case Operator::ATAN: {
      auto op = std::dynamic_pointer_cast<const OperatorNode>(node);
      auto x = map.at(op->in_nodes[0]);
      // {atan(k), k3}, // constant fold
      if (x->op == Operator::CONSTANT) {
        return std::make_shared<ConstantNode>(std::atan(k(x)));
      }
      if (x == op->in_nodes[0]) {
        return node;
      }
      return make_operator(Operator::ATAN, {x});
    }

    case Operator::LGAMMA: {
      auto op = std::dynamic_pointer_cast<const OperatorNode>(node);
      auto x = map.at(op->in_nodes[0]);
      // {lgamma(k), k3}, // constant fold
      if (x->op == Operator::CONSTANT) {
        return std::make_shared<ConstantNode>(std::lgamma(k(x)));
      }
      if (x == op->in_nodes[0]) {
        return node;
      }
      return make_operator(Operator::LGAMMA, {x});
    }

    case Operator::POLYGAMMA: {
      auto op = std::dynamic_pointer_cast<const OperatorNode>(node);
      auto left = map.at(op->in_nodes[0]);
      auto right = map.at(op->in_nodes[1]);
      // {polygamma(k1, k2), k3}, // constant fold
      if (left->op == Operator::CONSTANT && right->op == Operator::CONSTANT) {
        auto value = boost::math::polygamma(k(left), k(right));
        return std::make_shared<ConstantNode>(value);
      }
      if (left == op->in_nodes[0] && right == op->in_nodes[1]) {
        return node;
      }
      return make_operator(Operator::POLYGAMMA, {left, right});
    }

    default:
      break;
  }

  return node;
}

// The following method may be useful in debugging the problematic case
// that the rewrite_one method returns nested nodes that it does not
// place in the map. It is commented out in normal use, but when the
// optimizer throws an exception because a node is not in the map, this
// will likely be helpful in finding the problem.
bool check_children(const Nodep& node, NodeValueMap<Nodep>& map) {
  if (auto op = std::dynamic_pointer_cast<const OperatorNode>(node)) {
    for (auto& in : op->in_nodes) {
      if (!map.contains(in) || map.at(in) == nullptr) {
        return false;
      }
    }
  }
  return true;
}

// An intermediate method placed into the call-chain just to make debugging
// easier.
inline Nodep rewrite_one_internal(const Nodep& node, NodeValueMap<Nodep>& map) {
  return rewrite_one(node, map);
}

// Call the rewriter repeatedly on a node until a fixed-point is reached, and
// then place the result in the node-value-based map.
Nodep rewrite_node(const Nodep& node, NodeValueMap<Nodep>& map) {
  // check_children(node, map);
  Nodep rewritten = node;
  while (true) {
    const Nodep n = rewrite_one_internal(rewritten, map);
    if (same(n, rewritten)) {
      rewritten = n;
      break;
    }
    if (n == nullptr) {
      throw std::logic_error("rewriter should not return nullptr");
    }
    map[rewritten] = n;
    rewritten = n;
  }
  if (auto found = map.find(node);
      found == map.end() || !same(rewritten, found->second)) {
    if (rewritten == nullptr) {
      throw std::logic_error("rewriter should not return nullptr");
    }
    map[node] = rewritten;
  }
  map[rewritten] = rewritten;
  return rewritten;
}

std::unordered_map<Nodep, Nodep> opt_map(std::vector<Nodep> roots) {
  std::vector<Nodep> sorted;
  if (!topological_sort<Nodep>(
          {roots.begin(), roots.end()}, in_nodes, sorted)) {
    throw std::invalid_argument("graph has a cycle");
  }
  std::reverse(sorted.begin(), sorted.end());

  // a value-based, map, which treats semantically identical nodes as the same.
  NodeValueMap<Nodep> map;
  for (auto& node : sorted) {
    rewrite_node(node, map);
  }

  // We also build a map that uses object (pointer) identity to find elements,
  // so that operations in clients are not using recursive equality operations.
  std::unordered_map<Nodep, Nodep> identity_map;
  for (auto& node : sorted) {
    identity_map.insert({node, map.at(node)});
  }
  return identity_map;
}

} // namespace beanmachine::minibmg
