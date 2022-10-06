/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/format.h>
#include <memory>
#include <random>
#include "beanmachine/minibmg/distribution/distribution.h"
#include "beanmachine/minibmg/distribution/make_distribution.h"
#include "beanmachine/minibmg/eval_error.h"
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/node.h"
#include "beanmachine/minibmg/observations_by_node.h"

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

template <class T>
requires Number<T>
struct EvalResult {
  // The log probability of the overall computation.
  T log_prob;

  // The value of the queries.
  std::vector<double> queries;
};

template <class T>
requires Number<T>
struct SampledValue {
  T constrained;
  T unconstrained;
  T log_prob;
};

template <class T>
requires Number<T> SampledValue<T> sample_from_distribution(
    const Distribution<T>& distribution,
    std::mt19937& gen) {
  auto transformation = distribution.transformation();
  if (transformation == nullptr) {
    T constrained = distribution.sample(gen);
    T unconstrained = constrained;
    T log_prob = distribution.log_prob(constrained);
    return {constrained, unconstrained, log_prob};
  } else {
    T constrained = distribution.sample(gen);
    T unconstrained = transformation->call(constrained);
    T log_prob = distribution.log_prob(constrained);
    // Transforming the log_prob is on hold until I understand the math.
    // log_prob = transformation->transform_log_prob(constrained, log_prob);
    return {constrained, unconstrained, log_prob};
  }
}

// Evaluating an entire graph, producing into `data` a map of doubles that
// contains, for each scalar-valued node at graph index i, the evaluated value
// of that node at the corresponding index in the returned value.  Also returns
// the log probability of the samples.  The sampler function, if passed, is used
// to sample from the distribution.  It should return the sample in both
// constrained and unconstrained spaces, and a log_prob value with respect to
// its distribution transformed to the unconstrained space.
template <class T>
requires Number<T> EvalResult<T> eval_graph(
    const Graph& graph,
    std::mt19937& gen,
    std::function<T(const std::string& name, const unsigned identifier)>
        read_variable,
    std::unordered_map<Nodep, T>& data,
    bool run_queries = false,
    bool eval_log_prob = false,
    std::function<
        SampledValue<T>(const Distribution<T>& distribution, std::mt19937& gen)>
        sampler = sample_from_distribution<T>) {
  std::unordered_map<Nodep, std::shared_ptr<const Distribution<T>>>
      distributions;

  auto& obs_by_node = observations_by_node(graph);
  T log_prob = 0;
  for (const auto& node : graph) {
    switch (node->op) {
        // only need to handle SAMPLE here.
        // if a node is queried or reused, save it in data or distributions.
      case Operator::VARIABLE: {
        auto v = std::dynamic_pointer_cast<const VariableNode>(node);
        data[node] = read_variable(v->name, v->identifier);
        break;
      }
      case Operator::CONSTANT: {
        auto c = std::dynamic_pointer_cast<const ConstantNode>(node);
        data[node] = c->value;
        break;
      }
      case Operator::SAMPLE: {
        auto obsp = obs_by_node.find(node);
        auto sample_node = std::dynamic_pointer_cast<const SampleNode>(node);
        auto dist_node = sample_node->distribution;
        auto dist = distributions[dist_node];
        if (obsp != obs_by_node.end()) {
          auto value = data[node] = obsp->second;
          if (eval_log_prob) {
            T logp = dist->log_prob(value);
            log_prob = log_prob + logp;
          }
        } else {
          auto sampled_value = sampler(*dist, gen);
          data[node] = sampled_value.constrained;
          if (eval_log_prob) {
            log_prob = log_prob + sampled_value.log_prob;
          }
        }
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
            data[node] = eval_operator<T>(node->op, get_parameter);
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

  std::vector<double> queries;
  if (run_queries) {
    for (const auto& q : graph.queries) {
      auto d = data.find(q);
      double value = (d == data.end()) ? 0 : d->second.as_double();
      queries.push_back(value);
    }
  }
  return {log_prob, queries};
}

} // namespace beanmachine::minibmg
