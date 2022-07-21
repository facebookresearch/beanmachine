/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <folly/json.h>
#include <string>
#include <vector>
#include "beanmachine/minibmg/container.h"

using uint = unsigned int;

namespace beanmachine::minibmg {

class Graph;
class Node;
class OperatorNode;
class ConstantNode;

enum class Operator {
  // An operator value that indicates no operator.  Used as a flag to
  // reflect an invalid operator value.
  NO_OPERATOR,

  // A scalar constant, like 1.2.
  // Result: the given constant value (REAL)
  CONSTANT,

  // A scalar variable.  Used for symbolid auto-differentiation (AD).
  VARIABLE,

  // Add two scalars.
  // Result: the sum (REAL)
  ADD,

  // Subtract one scalar from another.
  // Result: the difference (REAL)
  SUBTRACT,

  // Negate a scalar.
  // Result: the negated value (REAL)
  NEGATE,

  // Multiply two scalars.
  // Result: the product (REAL)
  MULTIPLY,

  // Divide one scalar by another.
  // Result: the quotient (REAL)
  DIVIDE,

  // Raise on scalar to the power of another.
  // Result: REAL
  POW,

  // Raise e to the power of the given scalar.
  // Result: REAL
  EXP,

  // The natural logarithm of the given scalar.
  // Result: REAL
  LOG,

  // The arctangent (functional inverse of the tangent) of a scalar.
  // Result: REAL
  ATAN,

  // The lgamma function
  // Result: REAL
  LGAMMA,

  // The polygamma(x, n) function.  polygamma(x, 0) is also known as digamma(x)
  // Note the order of parameters.
  // Result: REAL
  POLYGAMMA,

  // If the first argument is equal to the second, yields the third, else the
  // fourth.
  // Result: REAL
  IF_EQUAL,

  // If the first argument is less than the second, yields the third, else the
  // fourth.
  // Result: REAL
  IF_LESS,

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

  // A bernoulli distribution (DISTRIBUTION).
  // Parameters:
  // - probability of yeilding 1 (as opposed to 0) (REAL)
  // Result: the distribution (DISTRIBUTION)
  DISTRIBUTION_BERNOULLI,

  // Draw a sample from the distribution parameter.
  // Parameters:
  // - ditribution
  // Result: A sample from the distribution (REAL)
  SAMPLE,

  // Observe a sample from the distribution parameter.
  // Parameters:
  // - ditribution (DISTRIBUTION)
  // - value (REAL)
  // Result: NONE.
  OBSERVE,

  // Query an intermediate result in the graph.
  // Parameters:
  // - value (REAL)
  // Result: NONE.
  QUERY,

  // Not a real operator.  Used as a limit when looping through operators.
  LAST_OPERATOR,
};

Operator operator_from_name(const std::string& name);
std::string to_string(Operator op);
uint arity(Operator op);

enum class Type {
  // No type.  For example, the result of an observation or query node.
  NONE,

  // A scalar real value.
  REAL,

  // A distribution of real values.
  DISTRIBUTION,

  // Not a real type.  Used as a limit when looping through types.
  LAST_TYPE,
};

Type type_from_name(const std::string& name);
std::string to_string(Type type);

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
  inline const Node* operator[](uint node_id) const {
    return nodes[node_id];
  }

 private:
  const std::vector<const Node*> nodes;

  // A private constructor that forms a graph without validation.
  // Used internally.  All exposed graphs should be validated.
  explicit Graph(std::vector<const Node*> nodes);
  static void validate(std::vector<const Node*> nodes);

 public:
  class Factory {
   public:
    uint add_constant(double value);

    uint add_operator(enum Operator op, std::vector<uint> parents);

    // returns the index of the query in the samples
    uint add_query(uint parent);

    uint add_variable(const std::string& name, const uint variable_index);

    inline const Node* operator[](uint node_id) const {
      return nodes[node_id];
    }
    Graph build();
    ~Factory();

   private:
    std::vector<const Node*> nodes;
    uint next_query = 0;
  };
};

// Exception to throw when json_to_graph fails.
class JsonError : public std::exception {
 public:
  explicit JsonError(const std::string& message);
  const std::string message;
};

folly::dynamic graph_to_json(const Graph& g);
Graph json_to_graph(folly::dynamic d); // throw (JsonError)

class Node {
 public:
  Node(const uint sequence, const enum Operator op, const Type type);
  const uint sequence;
  const enum Operator op;
  const enum Type type;
  virtual ~Node() = 0;
};

class OperatorNode : public Node {
 public:
  OperatorNode(
      const std::vector<const Node*>& in_nodes,
      const uint sequence,
      const enum Operator op,
      const enum Type type);
  const std::vector<const Node*> in_nodes;
};

class ConstantNode : public Node {
 public:
  ConstantNode(const double value, const uint sequence);
  const double value;
};

class VariableNode : public Node {
 public:
  VariableNode(
      const std::string& name,
      const uint variable_index,
      const uint sequence);
  const std::string name;
  const uint variable_index;
};

class QueryNode : public Node {
 public:
  QueryNode(const uint query_index, const Node* in_node, const uint sequence);
  const uint query_index;
  const Node* const in_node;
};

} // namespace beanmachine::minibmg
