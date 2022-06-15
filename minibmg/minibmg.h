/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace beanmachine::minibmg {

class GraphFactory {
 public:
  uint add_constant(double value);
  uint add_operator(OperatorType op, std::vector<uint> parents);
  Node* get_node(uint node_id);
  Graph build();

 private:
  std::vector<Node*> nodes;
};

class Graph {
 public:
  const std::vector<graph::Node*>& nodes;
};

class Node {
 public:
  const uint sequence;
  const Operator operator;
  const Type type;
};

class OperatorNode : Node {
 public:
  const std::vector<graph::Node*>& in_nodes;
}

class ConstantNode : Node {
 public:
  const double value;
};

class QueryNode : OperatorNode {
 public:
  const uint query_index;
}

enum Operator {
  // A scalar constant, like 1.2
  // Result: the given constant value (REAL)
  CONSTANT,
  // A normal distribution.
  // Parameters:
  // - mean (REAL)
  // - standard deviation (REAL)
  // Result: the distribution (DISTRIBUTION)
  DISTRIBUTION_NORMAL,
  // A beta distribution.
  // Parameters:
  // - ??? (REAL)
  // - ??? (REAL)
  // Result: the distribution.
  DISTRIBUTION_BETA,
  // A bernoulli distribution (DISTRIBUTION)
  // Parameters:
  // - probability of yeilding 1 (as opposed to 0) (REAL)
  // Result: the distribution (DISTRIBUTION)
  DISTRIBUTION_BERNOULLI,
  // Draw a sample from the distribution parameter.
  // Parameters:
  // - ditribution
  // Result: the distribution (DISTRIBUTION)
  SAMPLE,
  // Observe a sample from the distribution parameter.
  // Parameters:
  // - ditribution (DISTRIBUTION)
  // - value (REAL)
  // Result: The given value (NONE)
  OBSERVE,
  // Query an intermediate result in the graph.
  // Parameters:
  // - value (REAL)
  // Result: NONE.
  QUERY,
};

enum Type {
  // No type.  For example, the result of an observation node.
  NONE,
  // A scalar real value.
  REAL,
  // A distribution of real values.
  DISTRIBUTION,
};

} // namespace beanmachine::minibmg
