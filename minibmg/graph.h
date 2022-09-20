/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <folly/json.h>
#include "beanmachine/minibmg/container.h"
#include "beanmachine/minibmg/node.h"

namespace beanmachine::minibmg {

class Graph : public Container {
 public:
  // valudates that the list of nodes forms a valid graph,
  // and returns that graph.  Throws an exception if the
  // nodes do not form a valid graph.
  static Graph create(std::vector<const Node*> nodes);
  ~Graph();

  // Implement the iterator pattern so clients can iterate over the nodes.
  inline auto begin() const {
    return nodes.begin();
  }
  inline auto end() const {
    return nodes.end();
  }
  inline const Node* operator[](int index) const {
    return nodes[index];
  }
  inline int size() const {
    return nodes.size();
  }

  inline const Node* operator[](NodeId node_id) const {
    return nodes[node_id];
  }

 private:
  const std::vector<const Node*> nodes;

  // A private constructor that forms a graph without validation.
  // Used internally.  All exposed graphs should be validated.
  explicit Graph(std::vector<const Node*> nodes);
  static void validate(std::vector<const Node*> nodes);

 public:
  class Factory;
  class FluentFactory;
};

// Exception to throw when json_to_graph fails.
class JsonError : public std::exception {
 public:
  explicit JsonError(const std::string& message);
  const std::string message;
};

folly::dynamic graph_to_json(const Graph& g);
Graph json_to_graph(folly::dynamic d); // throw (JsonError)

} // namespace beanmachine::minibmg
