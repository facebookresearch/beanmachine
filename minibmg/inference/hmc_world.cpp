/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/inference/hmc_world.h"
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/ad/reverse.h"
#include "beanmachine/minibmg/ad/traced.h"
#include "beanmachine/minibmg/distribution/bernoulli.h"
#include "beanmachine/minibmg/eval.h"
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/graph_properties/observations_by_node.h"
#include "beanmachine/minibmg/graph_properties/unobserved_samples.h"
#include "beanmachine/minibmg/node.h"
#include "beanmachine/minibmg/pretty.h"
#include "beanmachine/minibmg/rewriters/dedag.h"

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
      assert(!result.empty());
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

  auto& queries = eval_result.queries;
  int n = queries.size();
  result.resize(n);
  for (int i = 0; i < n; i++) {
    assert(!result.empty());
    result[i] = queries[i].as_double();
  }
}

// "In a world in which" we evaluate things by computing them the hard way once,
// symbolically, and then optimize and save the symbolic form for (hopefully)
// fast recursive evaluation later.  This reduces the memory allocation
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

// An implementation of HMCWorld that compultes things symbolically, once, up
// front, and then evaluates that symbolic form when required later.
HMCWorld1::HMCWorld1(const Graph& graph)
    : graph{graph},
      unobserved_sample_set{
          unobserved_samples(graph).begin(),
          unobserved_samples(graph).end()},
      observations{observations_by_node(graph)} {
  //
  // We evaluate the graph using automatic derivatives (AD) in reverse mode,
  // computing symbolic forms for the queries, log_prob, and gradients.  We save
  // those symbolic forms for fast recursive evaluation later.
  //
  using T = Reverse<Traced>;

  // We count the unobserved samples as we encounter them, and save them for
  // later extraction of the gradient computed in reverse mode AD.
  unsigned next_sample = 0;
  std::vector<T> samples{};

  // Here is our function for producing a value for an unobserved sample.  Since
  // we are building an expression tree and not actually evaluating this code
  // numerically when it is executed, we create a fresh variable for each
  // proposed value. Then we use that variable in the resulting expression tree.
  // Later, when the resulting tree is evaluated numerically, we "read the value
  // of a variable" by pulling a value out of the vector of numeric proposed
  // values.
  std::function<SampledValue<T>(
      const Distribution<T>& distribution, std::mt19937& gen)>
      sample_from_distribution = [&](const Distribution<T>& distribution,
                                     std::mt19937&) -> SampledValue<T> {
    auto this_sample = next_sample++;
    auto variable_name = fmt::format("proposals[{}]", this_sample);
    auto variable_identifier = this_sample;

    // Here we produce a fresh variable for each unobserved sample
    T unconstrained = T{Traced::variable(variable_name, variable_identifier)};

    // We keep track of all of the variables so we can use reverse-mode AD to
    // compute derivatives with respect to them.
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
  std::mt19937 random;
  std::unordered_map<Nodep, T> data{};
  auto eval_result = eval_graph<T>(
      graph,
      random,
      read_variable,
      data,
      /* run_queries = */ true,
      /* eval_log_prob  = */ true,
      sample_from_distribution);

  // Save the symbolic form of the computation of the log_prob
  std::vector<ScalarNodep> log_prob = {eval_result.log_prob.ptr->primal.node};

  // Save the symbolic form of the computation of the set of queries.
  std::vector<ScalarNodep> queries{};
  for (auto& q : eval_result.queries) {
    queries.push_back(q.ptr->primal.node);
  }

  // trigger the reverse pass of reverse-mode AD, using the log_prob as the
  // value that we want the derivative of with respect to each proposed sample.
  // This populates all of the reverse AD nodes with an adjoint value that is
  // the partial derivative of the log_prob with respect to the value stored in
  // that node.  Since we are working symbolically, the adjoint values are
  // expression dags.
  eval_result.log_prob.reverse(1);

  // Save the symbolic form of the computation of the gradient of the log_prob
  // value with respect to each of the unobserved samples.  These derivatives
  // were computed during the reverse pass of reverse-mode AD, just above.
  std::vector<ScalarNodep> gradients{};
  for (auto& g : samples) {
    // g is a Reverse<Traced>; Reverse is  a wrapper around a pointer (ptr)
    // to the reverse body.  The adjoint field of that contains a Traced for the
    // derivative that was computed during the reverse pass, just above.  The
    // node field of a Traced contains the Nodep for that expression dag.  We
    // capture that expression dag for later evaluation.
    gradients.push_back(g.ptr->adjoint.node);
  }

  // We optimize each of the saved symbolic forms and prepare them for fast
  // evaluation.

  // set print_optimized_code to true to print out the optimized code for
  // debugging purposes.  This is useful to see what further optimization
  // opportunities might exist.
  const bool print_optimized_code = false;

  this->log_prob_graph = dedag(opt(log_prob));
  if (print_optimized_code) {
    std::cout << "\ncode for optimized log_prob:\n";
    for (auto& p : this->log_prob_graph.prelude) {
      std::cout << " " << p.first->name << " = " << to_string(p.second)
                << std::endl;
    }
    for (auto& p : this->log_prob_graph.result) {
      const Nodep& q = p;
      std::cout << "   " << to_string(q) << std::endl;
    }
  }

  this->gradients_graph = dedag(opt(gradients));
  if (print_optimized_code) {
    std::cout << "\ncode for optimized gradients:\n";
    for (auto& p : this->gradients_graph.prelude) {
      std::cout << " " << p.first->name << " = " << to_string(p.second)
                << std::endl;
    }
    for (auto& p : this->gradients_graph.result) {
      const Nodep& q = p;
      std::cout << "   " << to_string(q) << std::endl;
    }
  }

  this->queries_graph = dedag(opt(queries));
  if (print_optimized_code) {
    std::cout << "\ncode for optimized queries:\n";
    for (auto& p : this->queries_graph.prelude) {
      std::cout << " " << p.first->name << " = " << to_string(p.second)
                << std::endl;
    }
    for (auto& p : this->queries_graph.result) {
      const Nodep& q = p;
      std::cout << "   " << to_string(q) << std::endl;
    }
    std::cout << std::endl;
  }
}

unsigned HMCWorld1::num_unobserved_samples() const {
  return unobserved_sample_set.size();
}

// A helper function to evaluate a saved symbolic expression graph (dedagged)
// which has variables referencing the array of proposed unconstrained variables
// (proposed_unconstrained_values).  Each variable with a non-negative
// identifier is used to pull a value out of the corresponding element of the
// array of proposed values.  Variables with negative identifiers represent
// temporary values computed during the "prelude" (see dedag.h).
void eval_saved_dedagged(
    const std::vector<double>& proposed_unconstrained_values,
    Dedagged<std::vector<ScalarNodep>> dedagged,
    std::vector<double>& result) {
  // TODO: can we share this temp vector among all invocations of this method
  // for the life of this World object?
  std::vector<double> temps;
  temps.resize(dedagged.prelude.size());
  std::function<double(const std::string&, const int)> read_variable =
      [&](const std::string&, const int identifier) -> double {
    return (identifier >= 0) ? proposed_unconstrained_values[identifier]
                             : temps[~identifier];
  };
  RecursiveNodeEvaluatorVisitor evaluator{read_variable};
  for (int i = 0, n = dedagged.prelude.size(); i < n; i++) {
    assert(dedagged.prelude[i].first->identifier == ~i);
    assert(!temps.empty());
    temps[i] =
        eval_node(evaluator, downcast<ScalarNode>(dedagged.prelude[i].second));
  }
  result.resize(dedagged.result.size());
  auto& dedagged_result = dedagged.result;
  for (int i = 0, n = dedagged_result.size(); i < n; i++) {
    assert(!result.empty());
    assert(!dedagged_result.empty());
    result[i] = eval_node(evaluator, dedagged_result[i]);
  }
}

double HMCWorld1::log_prob(
    const std::vector<double>& proposed_unconstrained_values) const {
  // TODO: share the same result vector from call to call.
  std::vector<double> result;
  eval_saved_dedagged(proposed_unconstrained_values, log_prob_graph, result);
  assert(!result.empty());
  return result[0];
}

void HMCWorld1::gradients(
    const std::vector<double>& proposed_unconstrained_values,
    std::vector<double>& result) const {
  eval_saved_dedagged(proposed_unconstrained_values, gradients_graph, result);
}

void HMCWorld1::queries(
    const std::vector<double>& proposed_unconstrained_values,
    std::vector<double>& result) const {
  eval_saved_dedagged(proposed_unconstrained_values, queries_graph, result);
}

} // namespace

namespace beanmachine::minibmg {

std::unique_ptr<const HMCWorld> hmc_world_0(const Graph& graph) {
  return std::make_unique<HMCWorld0>(graph);
}

std::unique_ptr<const HMCWorld> hmc_world_1(const Graph& graph) {
  return std::make_unique<HMCWorld1>(graph);
}

std::unique_ptr<const HMCWorld> hmc_world_2(const Graph&) {
  throw std::logic_error("hmc_world_2 not implemented");
}

} // namespace beanmachine::minibmg
