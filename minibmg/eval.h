/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/format.h>
#include <random>
#include "beanmachine/minibmg/distribution/distribution.h"
#include "beanmachine/minibmg/distribution/make_distribution.h"
#include "beanmachine/minibmg/eval_error.h"
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/node.h"

namespace {

using namespace beanmachine::minibmg;

template <class T>
T get(const std::unordered_map<Nodep, T>& map, Nodep id) {
  auto t = map.find(id);
  if (t == map.end()) {
    throw EvalError(fmt::format("Missing data for node"));
  }
  return t->second;
}

template <class T>
void put(std::unordered_map<Nodep, T>& map, Nodep id, const T& value) {
  map[id] = value;
}

} // namespace

namespace beanmachine::minibmg {

template <class N>
requires Number<N> N
eval_operator(Operator op, std::function<N(unsigned)> get_value) {
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

// Evaluating an entire graph, returning an array of doubles that contains, for
// each scalar-valued node at graph index i, the evaluated value of that node at
// the corresponding index in the returned value.  The caller is responsible for
// freeing the returned array.
template <class T>
requires Number<T>
void eval_graph(
    const Graph& graph,
    std::mt19937& gen,
    std::function<T(const std::string& name, const unsigned identifier)>
        read_variable,
    std::unordered_map<Nodep, T>& data) {
  // Copy observations into a map for easy access.
  std::unordered_map<Nodep, double> observations;
  std::unordered_map<Nodep, std::shared_ptr<const Distribution<T>>>
      distributions;
  for (auto p : graph.observations) {
    observations[p.first] = p.second;
  }
  int n = graph.size();
  for (int i = 0; i < n; i++) {
    Nodep node = graph[i];
    switch (node->op) {
      case Operator::VARIABLE: {
        auto v = std::dynamic_pointer_cast<const VariableNode>(node);
        put(data, node, read_variable(v->name, v->identifier));
        break;
      }
      case Operator::CONSTANT: {
        auto c = std::dynamic_pointer_cast<const ConstantNode>(node);
        put(data, node, T{c->value});
        break;
      }
      case Operator::SAMPLE: {
        auto obsp = observations.find(node);
        double value;
        if (obsp != observations.end()) {
          value = obsp->second;
        } else {
          auto sample = std::dynamic_pointer_cast<const OperatorNode>(node);
          auto dist = distributions[sample->in_nodes[0]];
          value = dist->sample(gen);
        }
        put(data, node, T{value});
        break;
      }
      case Operator::NO_OPERATOR: {
        throw EvalError("eval_graph does not support " + to_string(node->op));
      }
      default: {
        auto opnode = std::dynamic_pointer_cast<const OperatorNode>(node);
        std::function<T(unsigned)> get_parameter = [&](unsigned i) {
          return data[opnode->in_nodes[i]];
        };
        switch (node->type) {
          case Type::DISTRIBUTION: {
            distributions[node] = make_distribution(node->op, get_parameter);
            break;
          }
          case Type::REAL: {
            T result = eval_operator<T>(node->op, get_parameter);
            put(data, node, result);
            break;
          }
          default: {
            throw EvalError(
                "eval_graph does not support " + to_string(node->op));
          }
        }
      }
    }
  }
}

} // namespace beanmachine::minibmg
