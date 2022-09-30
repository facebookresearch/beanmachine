/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/dedup.h"
#include <map>
#include <stdexcept>
#include <unordered_map>
#include "beanmachine/minibmg/operator.h"
#include "beanmachine/minibmg/topological.h"

namespace {

using namespace beanmachine::minibmg;

// Provide a good hash function so Nodep values can be used in unordered maps
// and sets.  This treats Nodep values as semantically value-based.
struct NodepIdentityHash {
  std::size_t operator()(beanmachine::minibmg::Nodep const& p) const noexcept;
};

// Provide a good equality function so Nodep values can be used in unordered
// maps and sets.  This treats Nodep values, recursively, as semantically
// value-based.
struct NodepIdentityEquals {
  bool operator()(
      const beanmachine::minibmg::Nodep& lhs,
      const beanmachine::minibmg::Nodep& rhs) const noexcept;
};

// A value-based map from nodes to T.  Used for deduplicating a graph.
template <class T>
using NodeValueMap =
    std::unordered_map<Nodep, T, NodepIdentityHash, NodepIdentityEquals>;

// A value-based set of nodes.
using NodeValueSet =
    std::unordered_set<Nodep, NodepIdentityHash, NodepIdentityEquals>;

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
  // equal nodes have equal hash codes and equal operators.
  if (l == nullptr || r == nullptr ||
      l->cached_hash_value != r->cached_hash_value || l->op != r->op) {
    return false;
  }
  switch (l->op) {
    case Operator::VARIABLE: {
      const VariableNode* vl = dynamic_cast<const VariableNode*>(l);
      const VariableNode* vr = dynamic_cast<const VariableNode*>(r);
      return vl->name == vr->name && vl->identifier == vr->identifier;
    }
    case Operator::CONSTANT: {
      double cl = dynamic_cast<const ConstantNode*>(l)->value;
      double cr = dynamic_cast<const ConstantNode*>(r)->value;
      return std::isnan(cl) ? std::isnan(cr) : cl == cr;
    }
    case Operator::SAMPLE: {
      const SampleNode* sl = dynamic_cast<const SampleNode*>(l);
      const SampleNode* sr = dynamic_cast<const SampleNode*>(r);
      return sl->rvid == sr->rvid &&
          this->operator()(sl->distribution, sr->distribution);
    }
    default: {
      const OperatorNode* lo = dynamic_cast<const OperatorNode*>(l);
      const OperatorNode* ro = dynamic_cast<const OperatorNode*>(r);
      if (lo->in_nodes.size() != ro->in_nodes.size()) {
        return false;
      }
      auto it1 = lo->in_nodes.begin();
      auto it2 = ro->in_nodes.begin();
      for (; it1 != lo->in_nodes.end() && it2 != ro->in_nodes.end();
           it1++, it2++) {
        if (!this->operator()(*it1, *it2)) {
          return false;
        }
      }
      return true;
    }
  }
}

// Rewrite a single node by replacing all of its inputs with their deduplicated
// counterpart.
Nodep rewrite(Nodep node, const NodeValueMap<Nodep>& map) {
  switch (node->op) {
    case Operator::CONSTANT:
    case Operator::VARIABLE:
      return node;
    case Operator::SAMPLE: {
      auto s = std::dynamic_pointer_cast<const SampleNode>(node);
      Nodep dist = map.at(s->distribution);
      if (dist == s->distribution) {
        return node;
      }
      return std::make_shared<SampleNode>(dist, s->rvid);
    }
    default: {
      auto op = std::dynamic_pointer_cast<const OperatorNode>(node);
      std::vector<Nodep> in_nodes;
      bool changed = false;
      for (Nodep in_node : op->in_nodes) {
        Nodep replacement = map.at(in_node);
        if (replacement != in_node) {
          changed = true;
        }
        in_nodes.push_back(replacement);
      }
      if (!changed) {
        return node;
      }
      return std::make_shared<OperatorNode>(in_nodes, node->op, node->type);
    }
  }
}

} // namespace

namespace beanmachine::minibmg {

// Take a set of root nodes as input, and return a map of deduplicated nodes,
// which maps from a node in the transitive closure of the input to a
// corresponding node in the transitive closure of the deduplicated graph.
std::unordered_map<Nodep, Nodep> dedup(std::vector<Nodep> roots) {
  // a value-based, map, which treats semantically identical nodes as the same.
  NodeValueMap<Nodep> map;

  // We also build a map that uses object (pointer) identity to find elements,
  // so that operations in clients are not using recursive equality operations.
  std::unordered_map<Nodep, Nodep> identity_map;

  std::vector<Nodep> sorted;
  if (!topological_sort<Nodep>(
          {roots.begin(), roots.end()}, in_nodes, sorted)) {
    throw std::invalid_argument("graph has a cycle");
  }
  std::reverse(sorted.begin(), sorted.end());
  for (auto node : sorted) {
    auto found = map.find(node);
    if (found != map.end()) {
      map.insert({node, found->second});
      identity_map.insert({node, found->second});
      continue;
    }
    auto rewritten = rewrite(node, map);
    map.insert({node, rewritten});
    identity_map.insert({node, rewritten});
  }

  return identity_map;
}

} // namespace beanmachine::minibmg
