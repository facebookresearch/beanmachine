// Copyright (c) Facebook, Inc. and its affiliates.

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/nmc.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/profiler.h"
#include "beanmachine/graph/proposer/default_initializer.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/util.h"

#include "beanmachine/graph/stepper/nmc_scalar_single_site_stepper.h"

namespace beanmachine {
namespace graph {

bool NMCScalarSingleSiteStepper::is_applicable_to(graph::Node* tgt_node) {
  return tgt_node->value.type.variable_type == VariableType::SCALAR;
}

void NMCScalarSingleSiteStepper::step(
    Node* tgt_node,
    const std::vector<Node*>& det_affected_nodes,
    const std::vector<Node*>& sto_affected_nodes) {
  graph->pd_begin(ProfilerEvent::NMC_STEP);
  // Implements a Metropolis-Hastings step using the NMC proposer.
  //
  // We are given an unobserved stochastic "target" node and we wish
  // to compute a new value for it. The basic idea of the algorithm is:
  //
  // * Save the current state of the graph. Only deterministic nodes need be
  //   saved because stochastic nodes values may are in principle
  //   compatible with any values of other nodes.
  // * Compute the probability of the current state.
  //   Note that we only need the probability of the immediate stochastic
  //   descendants of the target node, since those are the only ones
  //   whose probability changes when its value is changed
  //   (the probabilities of other nodes becomes irrelevant since
  //   it gets canceled out in the acceptable probability calculation,
  //   as explained below).
  // * Obtains the proposal distribution (old_prop) conditioned on
  //   target node's initial ('old') value.
  // * Propose a new value for the target node.
  // * Evaluate the probability of the proposed new state.
  //   Again, only immediate stochastic nodes need be considered.
  // * Obtains the proposal distribution (new_prop) conditioned on
  //   target node's new value.
  // * Accept or reject the proposed new value based on the
  //   Metropolis-Hastings acceptance probability:
  //          P(new state) * P_new_prop(old state | new state)
  //   min(1, ------------------------------------------------ )
  //          P(old state) * P_old_prop(new state | old state)
  //   but note how the probabilities for the states only need to include
  //   the immediate stochastic descendants since the distributions
  //   are factorized and the remaining stochastic nodes have
  //   their probabilities unchanged and cancel out.
  // * If we rejected it, restore the saved state.

  auto proposal_distribution_given_old_value = get_proposal_distribution(
      tgt_node, tgt_node->value, det_affected_nodes, sto_affected_nodes);

  NodeValue new_value = nmc->sample(proposal_distribution_given_old_value);

  nmc->revertibly_set_and_propagate(
      tgt_node, new_value, det_affected_nodes, sto_affected_nodes);

  double new_sto_affected_nodes_log_prob =
      nmc->compute_log_prob_of(sto_affected_nodes);

  auto proposal_distribution_given_new_value = get_proposal_distribution(
      tgt_node, new_value, det_affected_nodes, sto_affected_nodes);

  NodeValue& old_value = nmc->get_old_value(tgt_node);
  double old_sto_affected_nodes_log_prob =
      nmc->get_old_sto_affected_nodes_log_prob();

  double logacc = new_sto_affected_nodes_log_prob -
      old_sto_affected_nodes_log_prob +
      proposal_distribution_given_new_value->log_prob(old_value) -
      proposal_distribution_given_old_value->log_prob(new_value);

  bool accepted = logacc > 0 or util::sample_logprob(nmc->gen, logacc);
  if (!accepted) {
    nmc->revert_set_and_propagate(tgt_node, det_affected_nodes);
  }

  // Gradients must be cleared (equal to 0)
  // at the end of each iteration.
  // TODO: the reason for that is not clear;
  // it should be possible to compute gradients
  // when needed without depending on them
  // being 0.
  // However, some code depends on this
  // but it is not clear where.
  // It would be good to identify these dependencies
  // and possibly remove the
  // dependency.
  // This was the case for example for
  // StochasticOperator::gradient_log_prob,
  // but that dependence has been removed.
  nmc->clear_gradients(det_affected_nodes);
  tgt_node->grad1 = 0;
  tgt_node->grad2 = 0;
  graph->pd_finish(ProfilerEvent::NMC_STEP);
}

// Returns the NMC proposal distribution conditioned on the
// target node's current value.
// NOTE: assumes that det_affected_nodes's values are already
// evaluated according to the target node's value.
std::unique_ptr<proposer::Proposer>
NMCScalarSingleSiteStepper::get_proposal_distribution(
    Node* tgt_node,
    NodeValue value,
    const std::vector<Node*>& det_affected_nodes,
    const std::vector<Node*>& sto_affected_nodes) {
  graph->pd_begin(ProfilerEvent::NMC_CREATE_PROP);

  tgt_node->grad1 = 1;
  tgt_node->grad2 = 0;
  nmc->compute_gradients(det_affected_nodes);

  double grad1 = 0;
  double grad2 = 0;
  for (Node* node : sto_affected_nodes) {
    node->gradient_log_prob(tgt_node, /* in-out */ grad1, /* in-out */ grad2);
  }

  // TODO: generalize so it works with any proposer, not just nmc_proposer:
  std::unique_ptr<proposer::Proposer> prop =
      proposer::nmc_proposer(value, grad1, grad2);
  graph->pd_finish(ProfilerEvent::NMC_CREATE_PROP);
  return prop;
}

} // namespace graph
} // namespace beanmachine
