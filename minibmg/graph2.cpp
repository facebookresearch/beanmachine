/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/graph2.h"
#include <list>
#include <memory>
#include <stdexcept>
#include <vector>
#include "beanmachine/minibmg/dedup2.h"
#include "beanmachine/minibmg/topological.h"

namespace {

using namespace beanmachine::minibmg;

const std::vector<Node2p> roots(
    const std::vector<Node2p>& queries,
    const std::list<std::pair<Node2p, double>>& observations) {
  std::list<Node2p> roots;
  for (auto& n : queries) {
    roots.push_back(n);
  }
  for (auto& p : observations) {
    if (!std::dynamic_pointer_cast<const ScalarSampleNode2>(p.first)) {
      throw std::invalid_argument(fmt::format("can only observe a sample"));
    }
    roots.push_front(p.first);
  }
  std::vector<Node2p> all_nodes;
  if (!topological_sort<Node2p>(roots, &in_nodes, all_nodes)) {
    throw std::invalid_argument("graph has a cycle");
  }
  std::reverse(all_nodes.begin(), all_nodes.end());
  return all_nodes;
}

struct QueriesAndObservations {
  std::vector<Node2p> queries;
  std::list<std::pair<Node2p, double>> observations;
  ~QueriesAndObservations() {}
};

} // namespace

namespace beanmachine::minibmg {

template <>
class DedupHelper2<QueriesAndObservations> {
 public:
  std::vector<Node2p> find_roots(const QueriesAndObservations& qo) const {
    std::vector<Node2p> roots;
    for (auto& q : qo.observations) {
      roots.push_back(q.first);
    }
    for (auto& n : qo.queries) {
      roots.push_back(n);
    }
    return roots;
  }
  QueriesAndObservations rewrite(
      const QueriesAndObservations& qo,
      const std::unordered_map<Node2p, Node2p>& map) const {
    DedupHelper2<std::vector<Node2p>> h1{};
    DedupHelper2<std::list<std::pair<Node2p, double>>> h2{};
    return QueriesAndObservations{
        h1.rewrite(qo.queries, map), h2.rewrite(qo.observations, map)};
  }
};

using dynamic = folly::dynamic;

Graph2 Graph2::create(
    const std::vector<Node2p>& queries,
    const std::list<std::pair<Node2p, double>>& observations) {
  for (auto& p : observations) {
    if (!std::dynamic_pointer_cast<const ScalarSampleNode2>(p.first)) {
      throw std::invalid_argument(fmt::format("can only observe a sample"));
    }
  }

  auto qo0 = QueriesAndObservations{queries, observations};
  auto qo1 = dedup2(qo0);

  std::vector<Node2p> all_nodes = roots(qo1.queries, qo1.observations);
  return Graph2{all_nodes, qo1.queries, qo1.observations};
}

Graph2::~Graph2() {}

Graph2::Graph2(
    const std::vector<Node2p>& nodes,
    const std::vector<Node2p>& queries,
    const std::list<std::pair<Node2p, double>>& observations)
    : nodes{nodes}, queries{queries}, observations{observations} {}

} // namespace beanmachine::minibmg
