/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <folly/json.h>
#include <list>
#include "beanmachine/minibmg/graph_properties/container.h"
#include "beanmachine/minibmg/node2.h"

namespace beanmachine::minibmg {

class Graph2 : public Container {
 public:
  // produces a graph by computing the transitive closure of the input queries
  // and observations and topologically sorting the set of nodes so reached.
  // valudates that the list of nodes so reached forms a valid graph, and
  // returns that graph.  Throws an exception if the nodes do not form a valid
  // graph.
  static Graph2 create(
      const std::vector<Node2p>& queries,
      const std::list<std::pair<Node2p, double>>& observations,
      std::unordered_map<Node2p, Node2p>* built_map = nullptr);
  ~Graph2();

  // Implement the iterator pattern so clients can iterate over the nodes.
  inline auto begin() const {
    return nodes.begin();
  }
  inline auto end() const {
    return nodes.end();
  }
  inline Node2p operator[](int index) const {
    return nodes[index];
  }
  inline int size() const {
    return nodes.size();
  }

  // All of the nodes, in a topologically sorted order such that a node can only
  // be used as an input in subsequent (and not previous) nodes.
  const std::vector<Node2p> nodes;

  // Queries of the model.  These are nodes whose values are sampled by
  // inference.
  const std::vector<Node2p> queries;

  // Observations of the model.  These are SAMPLE nodes in the model whose
  // values are known.
  const std::list<std::pair<Node2p, double>> observations;

 private:
  // A private constructor that forms a graph without validation.
  // Used internally.  All exposed graphs should be validated.
  Graph2(
      const std::vector<Node2p>& nodes,
      const std::vector<Node2p>& queries,
      const std::list<std::pair<Node2p, double>>& observations);

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

folly::dynamic graph_to_json(const Graph2& g);
Graph2 json_to_graph2(folly::dynamic d); // throw (JsonError)

} // namespace beanmachine::minibmg
