/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/inference/hmc_world.h"
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/ad/reverse.h"
#include "beanmachine/minibmg/ad/traced.h"
#include "beanmachine/minibmg/dedag.h"
#include "beanmachine/minibmg/distribution/bernoulli.h"
#include "beanmachine/minibmg/eval.h"
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/graph_properties/observations_by_node.h"
#include "beanmachine/minibmg/graph_properties/unobserved_samples.h"
#include "beanmachine/minibmg/node.h"

namespace {

using namespace beanmachine::minibmg;

template <class T, class U = double>
requires Number<T> EvalResult<T> evaluate_internal(
    const Graph& graph,
    const std::vector<U>& proposed_unconstrained_values,
    std::unordered_map<Nodep, T>& data,
    std::mt19937& gen,
    bool run_queries,
    bool eval_log_prob) {
  unsigned next_sample = 0;

  // Here is our function for producing an unobserved sample by drawing from
  // `proposed_unconstrained_values`.  We consume that data provided by the
  // caller, transforming it if necessary.
  std::function<SampledValue<T>(
      const Distribution<T>& distribution, std::mt19937& gen)>
      sample_from_distribution = [&](const Distribution<T>& distribution,
                                     std::mt19937&) -> SampledValue<T> {
    T unconstrained = proposed_unconstrained_values[next_sample++];
    auto transform = distribution.transformation();
    if (transform == nullptr) {
      const T& constrained = unconstrained;
      T logp = eval_log_prob ? distribution.log_prob(constrained) : 0;
      return {constrained, unconstrained, logp};
    } else {
      T constrained = transform->inverse(unconstrained);
      T logp = eval_log_prob ? distribution.log_prob(constrained) : 0;
      //
      // We avoid the transform here because it may change the location of the
      // highest likelihood value.  For example, with a beta(7, 5), the peak is
      // at X=0.6.  However, when viewed in the transformed space with the
      // log_prob value also transformed, the peak occurs at a value
      // corresponding to X=0.625.  I am probably misunderstanding the math.  I
      // need help understanding what to do here. For now we just avoid
      // transforming the log_prob value.
      //
      // // logp = transform->transform_log_prob(constrained, logp);
      return {constrained, unconstrained, logp};
    }
  };

  // we don't need to read "variables" from a graph because there is no
  // such concept as a variable in Bean Machine.
  auto read_variable = [](const std::string&, const unsigned) -> T {
    throw std::logic_error("Bean Machine models should not contain variables");
  };

  // evaluate the graph.
  return eval_graph<T>(
      graph,
      gen,
      read_variable,
      data,
      run_queries,
      eval_log_prob,
      sample_from_distribution);
}

// "In a world in which" we evaluate things by computing them the hard way,
// using automatic differentiation if necessary.
class HMCWorld0 : public HMCWorld {
 private:
  const Graph& graph;
  std::unordered_set<Nodep> unobserved_sample_set;
  std::unordered_map<Nodep, double> observations;

 public:
  explicit HMCWorld0(const Graph& graph);

  unsigned num_unobserved_samples() const override;

  double log_prob(
      const std::vector<double>& proposed_unconstrained_values) const override;

  void gradients(
      const std::vector<double>& proposed_unconstrained_values,
      std::vector<double>& result) const override;

  void queries(
      const std::vector<double>& proposed_unconstrained_values,
      std::vector<double>& result) const override;
};

HMCWorld0::HMCWorld0(const Graph& graph)
    : graph{graph},
      unobserved_sample_set{
          unobserved_samples(graph).begin(),
          unobserved_samples(graph).end()},
      observations{observations_by_node(graph)} {}

unsigned HMCWorld0::num_unobserved_samples() const {
  return unobserved_sample_set.size();
}

double HMCWorld0::log_prob(
    const std::vector<double>& proposed_unconstrained_values) const {
  using T = Real;
  std::unordered_map<Nodep, T> data;
  std::mt19937 gen;

  // evaluate the graph and its log_prob in normal mode
  auto eval_result = evaluate_internal<T>(
      graph,
      proposed_unconstrained_values,
      data,
      gen,
      /* run_queries = */ false,
      /* eval_log_prob = */ true);

  return eval_result.log_prob.as_double();
}

void HMCWorld0::gradients(
    const std::vector<double>& proposed_unconstrained_values,
    std::vector<double>& result) const {
  using T = Reverse<Real>;
  std::unordered_map<Nodep, T> data;
  std::mt19937 gen;

  // evaluate the graph and its log_prob in reverse mode
  auto eval_result = evaluate_internal<T>(
      graph,
      proposed_unconstrained_values,
      data,
      gen,
      /* run_queries = */ false,
      /* eval_log_prob = */ true);

  // extract the gradients for the unobserved samples.
  eval_result.log_prob.reverse(1);
  std::vector<double> gradients;
  auto& obs = observations_by_node(graph);
  result.resize(num_unobserved_samples());
  int i = 0;
  for (const auto& node : graph) {
    if (dynamic_cast<const ScalarSampleNode*>(node.get()) &&
        !obs.contains(node)) {
      // we found an unobserved sample.  Add its gradient to the result.
      auto found = data.find(node);
      // It is possible that the node is not found in the data.  This occurs
      // when the log_prob does not depend on that sample at all.  In this case
      // the gradient is zero.
      auto grad =
          (found == data.end()) ? 0 : found->second.adjoint().as_double();
      result[i++] = grad;
    }
  }
}

void HMCWorld0::queries(
    const std::vector<double>& proposed_unconstrained_values,
    std::vector<double>& result) const {
  // In order to evaluate the queries, we evaluate the graph without computing
  // gradients.
  using T = Real;

  std::unordered_map<Nodep, T> data;
  std::mt19937 gen;

  // evaluate the graph and its log_prob using real numbers
  auto eval_result = evaluate_internal<T>(
      graph,
      proposed_unconstrained_values,
      data,
      gen,
      /* run_queries = */ true,
      /* eval_log_prob = */ false);

  int n = eval_result.queries.size();
  result.resize(n);
  for (int i = 0; i < n; i++) {
    result[i] = eval_result.queries[i].as_double();
  }
}

// "In a world in which" we evaluate things by computing them the hard way once,
// symbolically, and then optimize and save the symbolic form for (hopefully)
// fast recursive evaluation later.  This recuces the memory allocation
// overhead, because we do not need to allocate nodes for reverse AD.
class HMCWorld1 : public HMCWorld {
 private:
  const Graph& graph;
  std::unordered_set<Nodep> unobserved_sample_set;
  std::unordered_map<Nodep, double> observations;

  Dedagged<std::vector<ScalarNodep>> log_prob_graph;
  Dedagged<std::vector<ScalarNodep>> gradients_graph;
  Dedagged<std::vector<ScalarNodep>> queries_graph;

 public:
  explicit HMCWorld1(const Graph& graph);

  unsigned num_unobserved_samples() const override;

  double log_prob(
      const std::vector<double>& proposed_unconstrained_values) const override;

  void gradients(
      const std::vector<double>& proposed_unconstrained_values,
      std::vector<double>& result) const override;

  void queries(
      const std::vector<double>& proposed_unconstrained_values,
      std::vector<double>& result) const override;
};

HMCWorld1::HMCWorld1(const Graph& graph)
    : graph{graph},
      unobserved_sample_set{
          unobserved_samples(graph).begin(),
          unobserved_samples(graph).end()},
      observations{observations_by_node(graph)} {
  // We evaluate the graph in reverse mode, computing symbolic forms for the
  // queries, log_prob, and gradients.  We save those symbolic forms for fast
  // recursive evaluation later.
  using T = Reverse<Traced>;

  unsigned next_sample = 0;
  std::vector<T> samples{};

  // Here is our function for producing an unobserved sample by drawing from
  // `proposed_unconstrained_values`.  We produce a variable that will consume
  // that data provided by the caller, transforming it if necessary.
  std::function<SampledValue<T>(
      const Distribution<T>& distribution, std::mt19937& gen)>
      sample_from_distribution = [&](const Distribution<T>& distribution,
                                     std::mt19937&) -> SampledValue<T> {
    auto this_sample = next_sample++;
    auto variable_name = fmt::format("proposals[{}]", this_sample);
    auto variable_identifier = this_sample;
    T unconstrained = T{Traced::variable(variable_name, variable_identifier)};
    samples.push_back(unconstrained);
    auto transform = distribution.transformation();
    if (transform == nullptr) {
      const T& constrained = unconstrained;
      T logp = distribution.log_prob(constrained);
      return {constrained, unconstrained, logp};
    } else {
      T constrained = transform->inverse(unconstrained);
      T logp = distribution.log_prob(constrained);

      // I am confused about how transforms are really supposed to work.  I am
      // probably not understanding the math.  I need help understanding what to
      // do here. For now we just avoid transforming the log_prob value.

      // // logp = transform->transform_log_prob(constrained, logp);
      return {constrained, unconstrained, logp};
    }
  };

  // we don't need to read "variables" from a graph because there is no
  // such concept as a variable in Bean Machine.
  auto read_variable = [](const std::string&, const unsigned) -> T {
    throw std::logic_error("Bean Machine models should not contain variables");
  };

  // evaluate the graph.
  std::mt19937 gen;
  std::unordered_map<Nodep, T> data{};
  auto eval_result = eval_graph<T>(
      graph,
      gen,
      read_variable,
      data,
      /* run_queries = */ true,
      /* eval_log_prob  = */ true,
      sample_from_distribution);
  eval_result.log_prob.reverse(1);

  std::vector<ScalarNodep> log_prob = {eval_result.log_prob.ptr->primal.node};
  std::vector<ScalarNodep> queries{};
  for (auto& q : eval_result.queries) {
    queries.push_back(q.ptr->primal.node);
  }
  std::vector<ScalarNodep> gradients{};
  for (auto& g : samples) {
    gradients.push_back(g.ptr->adjoint.node);
  }

  this->log_prob_graph = dedag(opt(log_prob));
  this->gradients_graph = dedag(opt(gradients));
  this->queries_graph = dedag(opt(queries));
}

unsigned HMCWorld1::num_unobserved_samples() const {
  return unobserved_sample_set.size();
}

void eval_graph(
    const std::vector<double>& proposed_unconstrained_values,
    Dedagged<std::vector<ScalarNodep>> dedagged,
    std::vector<double>& result) {
  // TODO: can we share this temp vector among all invocations of this method
  // for the life of this World object?
  std::vector<double> temps;
  temps.resize(dedagged.prelude.size());
  std::function<double(const std::string& name, const int identifier)>
      read_variable =
          [&](const std::string& name, const int identifier) -> double {
    if (identifier >= 0)
      return proposed_unconstrained_values[identifier];
    else
      return temps[~identifier];
  };
  RecursiveNodeEvaluatorVisitor evaluator{read_variable};
  for (int i = 0, n = dedagged.prelude.size(); i < n; i++) {
    assert(dedagged.prelude[i].first->identifier == ~i);
    temps[i] = eval_node(
        evaluator,
        std::dynamic_pointer_cast<const ScalarNode>(
            dedagged.prelude[i].second));
  }
  result.resize(dedagged.result.size());
  for (int i = 0, n = dedagged.result.size(); i < n; i++) {
    result[i] = eval_node(evaluator, dedagged.result[i]);
  }
}

double HMCWorld1::log_prob(
    const std::vector<double>& proposed_unconstrained_values) const {
  // TODO: share the same result vector from call to call.
  std::vector<double> result;
  eval_graph(proposed_unconstrained_values, log_prob_graph, result);
  return result[0];
}

void HMCWorld1::gradients(
    const std::vector<double>& proposed_unconstrained_values,
    std::vector<double>& result) const {
  eval_graph(proposed_unconstrained_values, gradients_graph, result);
}

void HMCWorld1::queries(
    const std::vector<double>& proposed_unconstrained_values,
    std::vector<double>& result) const {
  eval_graph(proposed_unconstrained_values, queries_graph, result);
}

} // namespace

namespace beanmachine::minibmg {

std::unique_ptr<const HMCWorld> hmc_world_0(const Graph& graph) {
  return std::make_unique<HMCWorld0>(graph);
}

std::unique_ptr<const HMCWorld> hmc_world_1(const Graph& graph) {
  return std::make_unique<HMCWorld1>(graph);
}

std::unique_ptr<const HMCWorld> hmc_world_2(const Graph& graph) {
  throw std::logic_error("hmc_world_2 not implemented");
}

} // namespace beanmachine::minibmg
