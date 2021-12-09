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
#include "beanmachine/graph/mh.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/profiler.h"
#include "beanmachine/graph/proposer/default_initializer.h"
#include "beanmachine/graph/proposer/proposer.h"
#include "beanmachine/graph/util.h"

#include "beanmachine/graph/stepper/single_site/nmc_scalar_single_site_stepping_method.h"

namespace beanmachine {
namespace graph {

bool NMCScalarSingleSiteSteppingMethod::is_applicable_to(
    graph::Node* tgt_node) {
  return tgt_node->value.type.variable_type == VariableType::SCALAR;
}

ProfilerEvent NMCScalarSingleSiteSteppingMethod::get_step_profiler_event() {
  return ProfilerEvent::NMC_STEP;
}

// Returns the NMC proposal distribution conditioned on the
// target node's current value.
// NOTE: assumes that det_affected_nodes's values are already
// evaluated according to the target node's value.
std::unique_ptr<proposer::Proposer>
NMCScalarSingleSiteSteppingMethod::get_proposal_distribution(Node* tgt_node) {
  mh->graph->pd_begin(ProfilerEvent::NMC_CREATE_PROP);

  tgt_node->grad1 = 1;
  tgt_node->grad2 = 0;
  mh->compute_gradients(mh->get_det_affected_nodes(tgt_node));

  double grad1 = 0;
  double grad2 = 0;
  for (Node* node : mh->get_sto_affected_nodes(tgt_node)) {
    node->gradient_log_prob(tgt_node, /* in-out */ grad1, /* in-out */ grad2);
  }

  // TODO: generalize so it works with any proposer, not just nmc_proposer:
  std::unique_ptr<proposer::Proposer> prop =
      proposer::nmc_proposer(tgt_node->value, grad1, grad2);
  mh->graph->pd_finish(ProfilerEvent::NMC_CREATE_PROP);
  return prop;
}

} // namespace graph
} // namespace beanmachine
