/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/inference/hmc_world.h"
#include <memory>
#include <random>
#include <unordered_set>
#include "beanmachine/minibmg/ad/real.h"
#include "beanmachine/minibmg/ad/reverse.h"
#include "beanmachine/minibmg/distribution/distribution.h"
#include "beanmachine/minibmg/eval2.h"
#include "beanmachine/minibmg/graph2.h"
#include "beanmachine/minibmg/graph_properties/observations_by_node2.h"
#include "beanmachine/minibmg/node2.h"

namespace {

using namespace beanmachine::minibmg;

class HMCWorld0 : public HMCWorld {
 private:
  const Graph2& graph;
  std::unordered_set<Node2p> unobserved_samples;

 public:
  explicit HMCWorld0(const Graph2& graph);

  unsigned num_unobserved_samples() const override;

  HMCWorldEvalResult evaluate(
      const std::vector<double>& proposed_unconstrained_values) const override;

  std::vector<double> queries(
      const std::vector<double>& proposed_unconstrained_values) const override;
};

HMCWorld0::HMCWorld0(const Graph2& graph) : graph{graph} {
  // we identify the set of unobserved samples by...
  for (const auto& node : graph.nodes) {
    // ...counting the samples...
    if (dynamic_cast<const ScalarSampleNode2*>(node.get())) {
      unobserved_samples.insert(node);
    }
  }
  // and subtracting the observed ones.
  for (const auto& p : graph.observations) {
    unobserved_samples.erase(p.first);
  }
}

unsigned HMCWorld0::num_unobserved_samples() const {
  return unobserved_samples.size();
}

template <class T>
requires Number<T> EvalResult<T> evaluate_internal(
    const Graph2& graph,
    const std::vector<double>& proposed_unconstrained_values,
    std::unordered_map<Node2p, T>& data,
    std::mt19937& gen,
    bool run_queries,
    bool eval_log_prob) {
  unsigned next_sample = 0;

  // Here is our function for producing an unobserved sample.  We consume the
  // data provided by the caller, transforming it if necessary.
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
      // corresponding to X=0.625.  I need help understanding what to do here.
      // For now we just avoid transforming the log_prob value.
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

HMCWorldEvalResult HMCWorld0::evaluate(
    const std::vector<double>& proposed_unconstrained_values) const {
  using T = Reverse<Real>;
  std::unordered_map<Node2p, T> data;
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
  for (const auto& node : graph) {
    if (dynamic_cast<const ScalarSampleNode2*>(node.get()) &&
        !obs.contains(node)) {
      // we found an unobserved sample.  Add its gradient to the result.
      auto found = data.find(node);
      // It is possible that the node is not found in the data.  This occurs
      // when the log_prob does not depend on that sample at all.  In this case
      // the gradient is zero.
      auto grad =
          (found == data.end()) ? 0 : found->second.adjoint().as_double();
      gradients.push_back(grad);
    }
  }

  return HMCWorldEvalResult{
      eval_result.log_prob.as_double(), std::move(gradients)};
}

std::vector<double> HMCWorld0::queries(
    const std::vector<double>& proposed_unconstrained_values) const {
  // In order to evaluate the queries, we evaluate the graph without computing
  // gradients.
  using T = Real;

  std::unordered_map<Node2p, T> data;
  std::mt19937 gen;

  // evaluate the graph and its log_prob using real numbers
  auto eval_result = evaluate_internal<T>(
      graph,
      proposed_unconstrained_values,
      data,
      gen,
      /* run_queries = */ true,
      /* eval_log_prob = */ false);

  return eval_result.queries;
}

} // namespace

namespace beanmachine::minibmg {

std::unique_ptr<const HMCWorld> hmc_world_0(const Graph2& graph) {
  return std::make_unique<HMCWorld0>(graph);
}

} // namespace beanmachine::minibmg
