/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>
#include <functional>
#include <random>
#include "beanmachine/minibmg/ad/number.h"
#include "beanmachine/minibmg/minibmg.h"

namespace beanmachine::minibmg {

// Exception to throw when evaluation fails.
class EvalError : public std::exception {
 public:
  explicit EvalError(const std::string& message) : message{message} {}
  const std::string message;
};

template <class N>
requires Number<N> N
eval_operator(Operator op, std::function<N(uint)> get_value) {
  switch (op) {
    case Operator::ADD:
      return get_value(0) + get_value(1);
    case Operator::SUBTRACT:
      return get_value(0) - get_value(1);
    case Operator::NEGATE:
      return -get_value(0);
    case Operator::MULTIPLY:
      return get_value(0) * get_value(1);
    case Operator::DIVIDE:
      return get_value(0) / get_value(1);
    case Operator::POW:
      return pow(get_value(0), get_value(1));
    case Operator::EXP:
      return exp(get_value(0));
    case Operator::LOG:
      return log(get_value(0));
    case Operator::ATAN:
      return atan(get_value(0));
    case Operator::LGAMMA:
      return lgamma(get_value(0));
    case Operator::POLYGAMMA: {
      // Note that we discard the gradients of n and require it be a constant.
      return polygamma((int)get_value(0).as_double(), get_value(1));
    }
    default:
      throw EvalError(
          "eval_operator does not support operator " + to_string(op));
  }
}

// Sample from the given distribution.
double sample_distribution(
    Operator distribution,
    std::function<double(uint)> get_parameter,
    std::mt19937& gen);

// Evaluating an entire graph, returning an array of doubles that contains, for
// each scalar-valued node at graph index i, the evaluated value of that node at
// the corresponding index in the returned value.  The caller is responsible for
// freeing the returned array.
template <class T>
requires Number<T>
void eval_graph(
    const Graph& graph,
    std::mt19937& gen,
    std::function<T(const std::string& name, const uint sequence)>
        read_variable,
    std::vector<T>& data) {
  int n = graph.size();
  for (int i = 0; i < n; i++) {
    const Node* node = graph[i];
    if (node->sequence != i) {
      throw EvalError(
          fmt::format("node at index {0} is numbered {1}", i, node->sequence));
    }
    switch (node->op) {
      case Operator::VARIABLE: {
        const VariableNode* v = static_cast<const VariableNode*>(node);
        data[i] = read_variable(v->name, v->sequence);
        break;
      }
      case Operator::CONSTANT: {
        const ConstantNode* c = static_cast<const ConstantNode*>(node);
        data[i] = c->value;
        break;
      }
      case Operator::SAMPLE: {
        const OperatorNode* sample = static_cast<const OperatorNode*>(node);
        const Node* in0 = sample->in_nodes[0];
        const OperatorNode* dist = static_cast<const OperatorNode*>(in0);
        std::function<double(uint)> get_parameter = [=](uint i) {
          return data[dist->in_nodes[i]->sequence].as_double();
        };
        data[i] = sample_distribution(dist->op, get_parameter, gen);
        break;
      }
      case Operator::QUERY: {
        // We treat a query like a sample.
        const QueryNode* sample = static_cast<const QueryNode*>(node);
        const Node* in0 = sample->in_node;
        const OperatorNode* dist = static_cast<const OperatorNode*>(in0);
        std::function<double(uint)> get_parameter = [=](uint i) {
          return data[dist->in_nodes[i]->sequence].as_double();
        };
        data[i] = sample_distribution(dist->op, get_parameter, gen);
        break;
      }
      case Operator::OBSERVE:
        // OBSERVE has no result and has no effect during evaluation.
        break;
      case Operator::NO_OPERATOR:
        throw EvalError(
            "sample_distribution does not support " + to_string(node->op));
      case Operator::DISTRIBUTION_BERNOULLI:
      case Operator::DISTRIBUTION_BETA:
      case Operator::DISTRIBUTION_NORMAL:
        // Distributions have no effect during evaluation.  They are examined
        // by downstream nodes such as SAMPLE.  However, given that we do
        // have an abstract class `Distribution`, we could create the
        // distribution here (once we figure out where to store it).
        break;
      default:
        const OperatorNode* opnode = static_cast<const OperatorNode*>(node);
        std::function<T(uint)> get_parameter = [=](uint i) {
          return data[opnode->in_nodes[i]->sequence];
        };
        T result = eval_operator<T>(node->op, get_parameter);
        data[i] = result;
    }
  }
}

} // namespace beanmachine::minibmg
