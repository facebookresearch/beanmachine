/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/format.h>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <variant>
#include "beanmachine/minibmg/ad/number.h"
#include "beanmachine/minibmg/dedup.h"
#include "beanmachine/minibmg/distribution/bernoulli.h"
#include "beanmachine/minibmg/distribution/beta.h"
#include "beanmachine/minibmg/distribution/distribution.h"
#include "beanmachine/minibmg/distribution/half_normal.h"
#include "beanmachine/minibmg/distribution/normal.h"
#include "beanmachine/minibmg/eval_error.h"
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/graph_properties/observations_by_node.h"
#include "beanmachine/minibmg/node.h"

namespace beanmachine::minibmg {

template <class N>
requires Number<N>
struct SampledValue {
  N constrained;
  N unconstrained;
  N log_prob;
};

} // namespace beanmachine::minibmg

namespace {

using namespace beanmachine::minibmg;

template <class N>
requires Number<N>
class NodeEvaluatorVisitor : public NodeVisitor {
 public:
  // We intentionally leave the following methods abstract for the caller to
  // fill in
  void visit(const ScalarVariableNode* node) override = 0;
  void visit(const ScalarSampleNode* node) override = 0;
  virtual N evaluate_input(const ScalarNodep& node) = 0;
  virtual std::shared_ptr<const Distribution<N>> evaluate_input_distribution(
      const DistributionNodep& node) = 0;

  N result;
  N evaluate_scalar(ScalarNodep& node) {
    node->accept(*this);
    return result;
  }

  std::shared_ptr<Distribution<N>> dist_result;
  std::shared_ptr<Distribution<N>> evaluate_distribution(
      DistributionNodep& node) {
    node->accept(*this);
    return dist_result;
  }

  void visit(const ScalarConstantNode* node) override {
    result = node->constant_value;
  }

  void visit(const ScalarAddNode* node) override {
    result = evaluate_input(node->left) + evaluate_input(node->right);
  }
  void visit(const ScalarSubtractNode* node) override {
    result = evaluate_input(node->left) - evaluate_input(node->right);
  }
  void visit(const ScalarNegateNode* node) override {
    result = -evaluate_input(node->x);
  }
  void visit(const ScalarMultiplyNode* node) override {
    result = evaluate_input(node->left) * evaluate_input(node->right);
  }
  void visit(const ScalarDivideNode* node) override {
    result = evaluate_input(node->left) / evaluate_input(node->right);
  }
  void visit(const ScalarPowNode* node) override {
    result = pow(evaluate_input(node->left), evaluate_input(node->right));
  }
  void visit(const ScalarExpNode* node) override {
    result = exp(evaluate_input(node->x));
  }
  void visit(const ScalarLogNode* node) override {
    result = log(evaluate_input(node->x));
  }
  void visit(const ScalarAtanNode* node) override {
    result = atan(evaluate_input(node->x));
  }
  void visit(const ScalarLgammaNode* node) override {
    result = lgamma(evaluate_input(node->x));
  }
  void visit(const ScalarPolygammaNode* node) override {
    int nv = (int)evaluate_input(node->n).as_double();
    result = polygamma(nv, evaluate_input(node->x));
  }
  void visit(const ScalarLog1pNode* node) override {
    result = log1p(evaluate_input(node->x));
  }
  void visit(const ScalarIfEqualNode* node) override {
    result = if_equal(
        evaluate_input(node->a),
        evaluate_input(node->b),
        evaluate_input(node->c),
        evaluate_input(node->d));
  }
  void visit(const ScalarIfLessNode* node) override {
    result = if_less(
        evaluate_input(node->a),
        evaluate_input(node->b),
        evaluate_input(node->c),
        evaluate_input(node->d));
  }
  void visit(const DistributionNormalNode* node) override {
    dist_result = std::make_shared<Normal<N>>(
        evaluate_input(node->mean), evaluate_input(node->stddev));
  }
  void visit(const DistributionHalfNormalNode* node) override {
    dist_result = std::make_shared<HalfNormal<N>>(evaluate_input(node->stddev));
  }
  void visit(const DistributionBetaNode* node) override {
    dist_result = std::make_shared<Beta<N>>(
        evaluate_input(node->a), evaluate_input(node->b));
  }
  void visit(const DistributionBernoulliNode* node) override {
    dist_result = std::make_shared<Bernoulli<N>>(evaluate_input(node->prob));
  }
};

template <class N>
requires Number<N>
class OneNodeAtATimeEvaluatorVisitor : public NodeEvaluatorVisitor<N> {
  std::function<N(const std::string& name, const unsigned identifier)>
      read_variable;
  std::unordered_map<const Node*, double> observations;
  N& log_prob;
  std::unordered_map<Nodep, N>& data;
  std::unordered_map<Nodep, std::shared_ptr<const Distribution<N>>>&
      distributions;
  bool eval_log_prob;
  std::mt19937& gen;
  const std::function<SampledValue<N>(
      const Distribution<N>& distribution,
      std::mt19937& gen)>& sampler;

  static std::unordered_map<const Node*, double> make_observations_by_node(
      const Graph& graph) {
    std::unordered_map<const Node*, double> result;
    for (auto& p : observations_by_node(graph)) {
      result[p.first.get()] = p.second;
    }
    return result;
  }

 public:
  OneNodeAtATimeEvaluatorVisitor(
      const Graph& graph,
      std::function<N(const std::string& name, const unsigned identifier)>
          read_variable,
      std::unordered_map<Nodep, N>& data,
      std::unordered_map<Nodep, std::shared_ptr<const Distribution<N>>>&
          distributions,
      N& log_prob,
      bool eval_log_prob,
      std::mt19937& gen,
      const std::function<SampledValue<
          N>(const Distribution<N>& distribution, std::mt19937& gen)>& sampler)
      : read_variable{read_variable},
        observations{make_observations_by_node(graph)},
        log_prob{log_prob},
        data{data},
        distributions{distributions},
        eval_log_prob{eval_log_prob},
        gen{gen},
        sampler{sampler} {}

  void visit(const ScalarVariableNode* node) override {
    this->result = read_variable(node->name, node->identifier);
  }
  void visit(const ScalarSampleNode* node) override {
    auto obsp = observations.find(node);
    auto dist_node = node->distribution;
    auto dist = distributions.at(dist_node);
    if (obsp != observations.end()) {
      auto value = this->result = obsp->second;
      if (eval_log_prob) {
        N logp = dist->log_prob(value);
        log_prob = log_prob + logp;
      }
    } else {
      auto sampled_value = sampler(*dist, gen);
      this->result = sampled_value.constrained;
      if (eval_log_prob) {
        log_prob = log_prob + sampled_value.log_prob;
      }
    }
  }
  N evaluate_input(const ScalarNodep& node) override {
    return data.at(node);
  }
  std::shared_ptr<const Distribution<N>> evaluate_input_distribution(
      const DistributionNodep& node) override {
    return distributions.at(node);
  }
};

} // namespace

namespace beanmachine::minibmg {

template <class N>
requires Number<N>
struct EvalResult {
  // The log probability of the overall computation.
  N log_prob;

  // The value of the queries.
  std::vector<double> queries;
};

template <class N>
requires Number<N> SampledValue<N> sample_from_distribution(
    const Distribution<N>& distribution,
    std::mt19937& gen) {
  auto transformation = distribution.transformation();
  if (transformation == nullptr) {
    N constrained = distribution.sample(gen);
    N unconstrained = constrained;
    N log_prob = distribution.log_prob(constrained);
    return {constrained, unconstrained, log_prob};
  } else {
    N constrained = distribution.sample(gen);
    N unconstrained = transformation->call(constrained);
    N log_prob = distribution.log_prob(constrained);
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
template <class N>
requires Number<N> EvalResult<N> eval_graph(
    const Graph& graph,
    std::mt19937& gen,
    std::function<N(const std::string& name, const unsigned identifier)>
        read_variable,
    std::unordered_map<Nodep, N>& data,
    bool run_queries = false,
    bool eval_log_prob = false,
    std::function<
        SampledValue<N>(const Distribution<N>& distribution, std::mt19937& gen)>
        sampler = sample_from_distribution<N>) {
  std::unordered_map<Nodep, std::shared_ptr<const Distribution<N>>>
      distributions;
  N log_prob = 0;

  std::function<SampledValue<N>(
      const Distribution<N>& distribution, std::mt19937& gen)>& sampler2 =
      sampler;
  OneNodeAtATimeEvaluatorVisitor<N> evaluator{
      graph,
      read_variable,
      data,
      distributions,
      log_prob,
      eval_log_prob,
      gen,
      sampler2};

  for (const auto& node : graph) {
    if (auto dist_node =
            std::dynamic_pointer_cast<const DistributionNode>(node)) {
      std::shared_ptr<const Distribution<N>> dist =
          evaluator.evaluate_distribution(dist_node);
      distributions[node] = dist;
    } else if (
        auto expr_node = std::dynamic_pointer_cast<const ScalarNode>(node)) {
      N expr = evaluator.evaluate_scalar(expr_node);
      data[node] = expr;
    } else {
      throw std::logic_error("unexpected node");
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

template <class Underlying>
requires Number<Underlying>
class DedupAdapter<EvalResult<Underlying>> {
  DedupAdapter<Underlying> helper{};

 public:
  std::vector<Nodep> find_roots(const EvalResult<Underlying>& e) const {
    return helper.find_roots(e.log_prob);
  }
  EvalResult<Underlying> rewrite(
      const EvalResult<Underlying>& e,
      const std::unordered_map<Nodep, Nodep>& map) const {
    auto new_log_prob = helper.rewrite(e.log_prob, map);
    return {new_log_prob, e.queries};
  }
};

} // namespace beanmachine::minibmg
