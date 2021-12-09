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

#include "beanmachine/graph/stepper/single_site/nmc_dirichlet_gamma_single_site_stepping_method.h"

namespace beanmachine {
namespace graph {

bool NMCDirichletGammaSingleSiteSteppingMethod::is_applicable_to(
    graph::Node* tgt_node) {
  return tgt_node->value.type.variable_type == VariableType::COL_SIMPLEX_MATRIX;
}

/*
We treat the K-dimensional Dirichlet sample as K independent Gamma samples
divided by their x_sum. i.e. Let X_k ~ Gamma(alpha_k, 1), for k = 1, ..., K,
Y_k = X_k / x_sum(X), then (Y_1, ..., Y_K) ~ Dirichlet(alphas). We store Y in
the attribute value, and X in unconstrainted_value.
*/
void NMCDirichletGammaSingleSiteSteppingMethod::step(Node* tgt_node) {
  mh->graph->pd_begin(ProfilerEvent::NMC_STEP_DIRICHLET);

  const std::vector<Node*>& det_affected_nodes =
      mh->get_det_affected_nodes(tgt_node);

  // Cast needed to access fields such as unconstrained_value:
  auto sto_tgt_node = static_cast<oper::StochasticOperator*>(tgt_node);
  // @lint-ignore CLANGTIDY
  auto dirichlet_distribution_node = sto_tgt_node->in_nodes[0];
  auto param_node = dirichlet_distribution_node->in_nodes[0];

  uint K = static_cast<uint>(tgt_node->value._matrix.size());
  for (uint k = 0; k < K; k++) {
    double param_a_k = param_node->value._matrix.coeff(k);
    double x_sum = sto_tgt_node->unconstrained_value._matrix.sum();

    // save old values
    mh->save_old_values(det_affected_nodes);
    double old_x_k = sto_tgt_node->unconstrained_value._matrix.coeff(k);
    NodeValue old_x_k_value(AtomicType::POS_REAL, old_x_k);
    double old_sto_affected_nodes_log_prob =
        compute_sto_affected_nodes_log_prob(tgt_node, param_a_k, old_x_k_value);

    // get proposal given old value
    auto proposal_given_old_value = create_proposal_dirichlet_gamma(
        tgt_node, param_a_k, x_sum, old_x_k_value, k);

    // sample new value
    NodeValue new_x_k_value = mh->sample(proposal_given_old_value);

    // set new value
    *(sto_tgt_node->unconstrained_value._matrix.data() + k) =
        new_x_k_value._double;
    x_sum = sto_tgt_node->unconstrained_value._matrix.sum();
    sto_tgt_node->value._matrix =
        sto_tgt_node->unconstrained_value._matrix.array() / x_sum;

    // propagate new value
    mh->eval(det_affected_nodes);
    double new_sto_affected_nodes_log_prob =
        compute_sto_affected_nodes_log_prob(tgt_node, param_a_k, new_x_k_value);

    // obtain proposal given new value
    auto proposal_given_new_value = create_proposal_dirichlet_gamma(
        tgt_node, param_a_k, x_sum, new_x_k_value, k);

    // compute acceptance probability
    double logacc = new_sto_affected_nodes_log_prob -
        old_sto_affected_nodes_log_prob +
        proposal_given_new_value->log_prob(old_x_k_value) -
        proposal_given_old_value->log_prob(new_x_k_value);

    // decide acceptance
    bool accepted = util::flip_coin_with_log_prob(mh->gen, logacc);
    if (!accepted) {
      // revert
      mh->restore_old_values(det_affected_nodes);
      *(sto_tgt_node->unconstrained_value._matrix.data() + k) = old_x_k;
      x_sum = sto_tgt_node->unconstrained_value._matrix.sum();
      sto_tgt_node->value._matrix =
          sto_tgt_node->unconstrained_value._matrix.array() / x_sum;
    }

    // Gradients are must be cleared (equal to 0)
    // at the end of each iteration.
    // Some code relies on that to decide whether a node
    // is the one we are computing gradients with respect to.
    // TODO: identify code that depends on this, let it zero gradients
    // itself, and remove it from here since that's a long-distance,
    // implicit dependence that is hard to watch for.
    mh->clear_gradients(det_affected_nodes);
  } // k
  mh->graph->pd_finish(ProfilerEvent::NMC_STEP_DIRICHLET);
}

double
NMCDirichletGammaSingleSiteSteppingMethod::compute_sto_affected_nodes_log_prob(
    Node* tgt_node,
    double param_a_k,
    NodeValue x_k_value) {
  double logweight = 0;
  for (Node* node : mh->get_sto_affected_nodes(tgt_node)) {
    if (node == tgt_node) {
      double& x_k = x_k_value._double;
      // X_k ~ Gamma(param_a_k, 1)
      // PDF of Gamma(a, 1) is x^(a - 1)exp(-x)/gamma(a)
      // so log pdf(x) = log(x^(a - 1)) + (-x) - log(gamma(a))
      // = (a - 1)*log(x) - x - log(gamma(a))
      logweight += (param_a_k - 1.0) * std::log(x_k) - x_k - lgamma(param_a_k);
    } else {
      logweight += node->log_prob();
    }
  }
  return logweight;
}

std::unique_ptr<proposer::Proposer>
NMCDirichletGammaSingleSiteSteppingMethod::create_proposal_dirichlet_gamma(
    Node* tgt_node,
    double param_a_k,
    double x_sum,
    NodeValue x_k_value,
    uint k) {
  mh->graph->pd_begin(ProfilerEvent::NMC_CREATE_PROP_DIR);

  // Cast needed to access fields such as unconstrained_value:
  auto sto_tgt_node = static_cast<oper::StochasticOperator*>(tgt_node);

  // Prepare gradients
  // Grad1 = (dY_1/dX_k, dY_2/dX_k, ..., dY_K/X_k)
  // where dY_k/dX_k = (x_sum(X) - X_k)/x_sum(X)^2
  //       dY_j/dX_k = - X_j/x_sum(X)^2, for j != k
  // Grad2 = (d^2Y_1/dX^2_k, ..., d^2Y_K/X^2_k)
  // where d2Y_k/dX2_k = -2 * (x_sum(X) - X_k)/x_sum(X)^3
  //       d2Y_j/dX2_k = -2 * X_j/x_sum(X)^3
  sto_tgt_node->Grad1 =
      -sto_tgt_node->unconstrained_value._matrix.array() / (x_sum * x_sum);
  *(sto_tgt_node->Grad1.data() + k) += 1 / x_sum;
  sto_tgt_node->Grad2 = sto_tgt_node->Grad1 * (-2.0) / x_sum;

  // Propagate gradients
  mh->compute_gradients(mh->get_det_affected_nodes(tgt_node));

  // We want to compute the gradient of log prob with respect to x_k.
  // The probability is the product of the probabilities of x_k
  // times the probabilities of the stochastic affected nodes other than
  // the target node.
  // Therefore, the logarithm of that is the log prob of x_k
  // plus the log probs of the stochastic affected nodes other than
  // the target node.
  // For the stochastic affected nodes, the method gradient_log_prob
  // provides that because we have already computed dY/dx_k above,
  // and then propagated that using compute_gradients,
  // which takes dY/dx_k and propagates through deterministic
  // nodes, computing log(d S_i/dY * dY/d_x_k) for each
  // stochastic affected node S_i other than the target node.
  double grad1 = 0;
  double grad2 = 0;
  for (Node* node : mh->get_sto_affected_nodes(tgt_node)) {
    if (node == tgt_node) {
      double& x_k = x_k_value._double;
      // X_k ~ Gamma(param_a_k, 1)
      // PDF of Gamma(a, 1) is x^(a - 1)exp(-x)/gamma(a)
      // so log pdf(x) = log(x^(a - 1)) + (-x) - log(gamma(a))
      // = (a - 1)*log(x) - x - log(gamma(a))
      grad1 += (param_a_k - 1.0) / x_k - 1.0;
      grad2 += (1.0 - param_a_k) / (x_k * x_k);
    } else {
      node->gradient_log_prob(tgt_node, /* in-out */ grad1, /* in-out */ grad2);
    }
  }
  std::unique_ptr<proposer::Proposer> proposal =
      proposer::nmc_proposer(x_k_value, grad1, grad2);
  mh->graph->pd_finish(ProfilerEvent::NMC_CREATE_PROP_DIR);
  return proposal;
}

} // namespace graph
} // namespace beanmachine
