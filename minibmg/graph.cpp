/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/graph.h"
#include <memory>
#include <stdexcept>
#include <vector>
#include "beanmachine/minibmg/rewriters/dedup.h"
#include "beanmachine/minibmg/topological.h"

namespace {

using namespace beanmachine::minibmg;

const std::vector<Nodep> roots(
    const std::vector<Nodep>& queries,
    const std::vector<std::pair<Nodep, double>>& observations) {
  std::vector<Nodep> roots;
  for (auto& n : queries) {
    roots.push_back(n);
  }
  for (auto& p : observations) {
    if (!downcast<ScalarSampleNode>(p.first)) {
      throw std::invalid_argument(fmt::format("can only observe a sample"));
    }
    roots.push_back(p.first);
  }
  std::vector<Nodep> all_nodes;
  if (!topological_sort<Nodep>(roots, &in_nodes, all_nodes)) {
    throw std::invalid_argument("graph has a cycle");
  }
  std::reverse(all_nodes.begin(), all_nodes.end());
  return all_nodes;
}

struct QueriesAndObservations {
  std::vector<Nodep> queries;
  std::vector<std::pair<Nodep, double>> observations;
  ~QueriesAndObservations() {}
};

} // namespace

namespace beanmachine::minibmg {

template <>
class NodeRewriteAdapter<QueriesAndObservations> {
 public:
  std::vector<Nodep> find_roots(const QueriesAndObservations& qo) const {
    std::vector<Nodep> roots;
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
      const std::unordered_map<Nodep, Nodep>& map) const {
    NodeRewriteAdapter<std::vector<Nodep>> h1{};
    NodeRewriteAdapter<std::vector<std::pair<Nodep, double>>> h2{};
    return QueriesAndObservations{
        h1.rewrite(qo.queries, map), h2.rewrite(qo.observations, map)};
  }
};

Graph Graph::create(
    const std::vector<Nodep>& queries,
    const std::vector<std::pair<Nodep, double>>& observations,
    std::unordered_map<Nodep, Nodep>* built_map) {
  for (auto& p : observations) {
    if (!downcast<ScalarSampleNode>(p.first)) {
      throw std::invalid_argument(fmt::format("can only observe a sample"));
    }
  }

  auto qo0 = QueriesAndObservations{queries, observations};
  auto qo1 = dedup(qo0, built_map);

  std::vector<Nodep> all_nodes = roots(qo1.queries, qo1.observations);
  return Graph{all_nodes, qo1.queries, qo1.observations};
}

Graph::~Graph() {}

Graph::Graph(
    const std::vector<Nodep>& nodes,
    const std::vector<Nodep>& queries,
    const std::vector<std::pair<Nodep, double>>& observations)
    : nodes{nodes}, queries{queries}, observations{observations} {}

} // namespace beanmachine::minibmg
