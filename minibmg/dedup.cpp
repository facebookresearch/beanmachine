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
