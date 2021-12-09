/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/profiler.h"
#include "beanmachine/graph/proposer/default_initializer.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/util.h"

#include "beanmachine/graph/stepper/single_site/default_single_site_stepping_method.h"

namespace beanmachine {
namespace graph {

void DefaultSingleSiteSteppingMethod::step(Node* tgt_node) {
  mh->graph->pd_begin(get_step_profiler_event());
  // Implements a Metropolis-Hastings step using the MH proposer.
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

  auto proposal_given_old_value = get_proposal_distribution(tgt_node);

  NodeValue new_value = mh->sample(proposal_given_old_value);

  mh->revertibly_set_and_propagate(tgt_node, new_value);

  double new_sto_affected_nodes_log_prob =
      mh->compute_log_prob_of(mh->get_sto_affected_nodes(tgt_node));

  auto proposal_given_new_value = get_proposal_distribution(tgt_node);

  NodeValue& old_value = mh->get_old_value(tgt_node);
  double old_sto_affected_nodes_log_prob =
      mh->get_old_sto_affected_nodes_log_prob();

  double logacc = new_sto_affected_nodes_log_prob -
      old_sto_affected_nodes_log_prob +
      proposal_given_new_value->log_prob(old_value) -
      proposal_given_old_value->log_prob(new_value);

  bool accepted = util::flip_coin_with_log_prob(mh->gen, logacc);
  if (!accepted) {
    mh->revert_set_and_propagate(tgt_node);
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
  mh->clear_gradients_of_node_and_its_affected_nodes(tgt_node);

  mh->graph->pd_finish(get_step_profiler_event());
}

} // namespace graph
} // namespace beanmachine
