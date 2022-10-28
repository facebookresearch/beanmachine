/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <folly/json.h>
#include <vector>
#include "beanmachine/minibmg/dedup.h"
#include "beanmachine/minibmg/graph_properties/container.h"
#include "beanmachine/minibmg/node.h"

namespace beanmachine::minibmg {

class Graph : public Container {
 public:
  // produces a graph by computing the transitive closure of the input queries
  // and observations and topologically sorting the set of nodes so reached.
  // valudates that the list of nodes so reached forms a valid graph, and
  // returns that graph. Throws an exception if the nodes do not form a valid
  // graph.  The caller can optionally pass a pointer to a map in which to
  // receive a copy of a map from the original nodes to the nodes appearing in
  // the graph (after any common subexpression elimination or optimization).

  static Graph create(
      const std::vector<Nodep>& queries,
      const std::vector<std::pair<Nodep, double>>& observations,
      std::unordered_map<Nodep, Nodep>* built_map = nullptr);
  ~Graph();

  // Implement the iterator pattern so clients can iterate over the nodes.
  inline auto begin() const {
    return nodes.begin();
  }
  inline auto end() const {
    return nodes.end();
  }
  inline Nodep operator[](int index) const {
    return nodes[index];
  }
  inline int size() const {
    return nodes.size();
  }

  // All of the nodes, in a topologically sorted order such that a node can only
  // be used as an input in subsequent (and not previous) nodes.
  const std::vector<Nodep> nodes;

  // Queries of the model.  These are nodes whose values are sampled by
  // inference.
  const std::vector<Nodep> queries;

  // Observations of the model.  These are SAMPLE nodes in the model whose
  // values are known.
  const std::vector<std::pair<Nodep, double>> observations;

 private:
  // A private constructor that forms a graph without validation.
  // Used internally.  All exposed graphs should be validated.
  Graph(
      const std::vector<Nodep>& nodes,
      const std::vector<Nodep>& queries,
      const std::vector<std::pair<Nodep, double>>& observations);

 public:
  // A factory for making graphs, like the bmg API used by Beanstalk
  class Factory;

  // A more natural factory for making graphs, using operator overloading.
  class FluidFactory;
};

// Exception to throw when json_to_graph fails.
class JsonError2 : public std::exception {
 public:
  explicit JsonError2(const std::string& message);
  const std::string message;
};

folly::dynamic graph_to_json(const Graph& g);
Graph json_to_graph2(folly::dynamic d); // throw (JsonError)

template <>
class NodeRewriteAdapter<Graph> {
 public:
  std::vector<Nodep> find_roots(const Graph& graph) const {
    std::vector<Nodep> roots;
    for (auto& q : graph.observations) {
      roots.push_back(q.first);
    }
    for (auto& n : graph.queries) {
      roots.push_back(n);
    }
    return roots;
  }
  Graph rewrite(const Graph& qo, const std::unordered_map<Nodep, Nodep>& map)
      const {
    NodeRewriteAdapter<std::vector<Nodep>> h1{};
    NodeRewriteAdapter<std::vector<std::pair<Nodep, double>>> h2{};
    return Graph::create(
        h1.rewrite(qo.queries, map), h2.rewrite(qo.observations, map));
  }
};

} // namespace beanmachine::minibmg
