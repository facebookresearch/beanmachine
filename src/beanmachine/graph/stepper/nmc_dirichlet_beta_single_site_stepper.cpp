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

#include "beanmachine/graph/stepper/nmc_dirichlet_beta_single_site_stepper.h"

namespace beanmachine {
namespace graph {

bool NMCDirichletBetaSingleSiteStepper::is_applicable_to(
    graph::Node* tgt_node) {
  return tgt_node->value.type.variable_type ==
      VariableType::COL_SIMPLEX_MATRIX and
      tgt_node->value.type.rows == 2;
}

/*
We treat the K-dimensional Dirichlet sample as K independent Gamma samples
divided by their sum. i.e. Let X_k ~ Gamma(alpha_k, 1), for k = 1, ..., K,
Y_k = X_k / sum(X), then (Y_1, ..., Y_K) ~ Dirichlet(alphas). We store Y in
the attribute value, and X in unconstrainted_value.
*/
void NMCDirichletBetaSingleSiteStepper::step(
    Node* tgt_node,
    const std::vector<Node*>& det_nodes,
    const std::vector<Node*>& sto_nodes) {
  graph->pd_begin(ProfilerEvent::NMC_STEP_DIRICHLET);
  assert(tgt_node->value._matrix.size() == 2);
  auto src_node = static_cast<oper::StochasticOperator*>(tgt_node);
  // @lint-ignore CLANGTIDY
  auto param_a = src_node->in_nodes[0]->in_nodes[0]->value._matrix.coeff(0);
  auto param_b = src_node->in_nodes[0]->in_nodes[0]->value._matrix.coeff(1);
  double old_X_k;
  // Prepare gradients
  // Grad1 = (dY_1/dX_1, dY_2/dX_1)
  // where dY_1/dX_1 = 1
  //       dY_j/dX_k = -1
  // Grad2 = (d^2Y_1/dX^2_1, d^2Y_2/X^2_1)
  // where d2Y_k/dX2_k = 0
  //       d2Y_j/dX2_k = 0
  old_X_k = src_node->value._matrix.coeff(0);
  Eigen::MatrixXd Grad1(2, 1);
  Grad1 << 1, -1;
  src_node->Grad1 = Grad1;
  *(src_node->Grad1.data() + 1) = -1;
  src_node->Grad2 = Eigen::MatrixXd::Zero(2, 1);
  src_node->grad1 = 1;
  src_node->grad2 = 0;

  // Propagate gradients
  NodeValue old_value(AtomicType::PROBABILITY, old_X_k);
  nmc->save_old_values(det_nodes);
  nmc->compute_gradients(det_nodes);
  double old_sto_affected_nodes_log_prob;
  auto old_prop = create_proposer_dirichlet_beta(
      sto_nodes,
      tgt_node,
      param_a,
      param_b,
      old_value,
      /* out */ old_sto_affected_nodes_log_prob);

  NodeValue new_value = nmc->sample(old_prop);
  *(src_node->value._matrix.data()) = new_value._double;
  *(src_node->value._matrix.data() + 1) = 1 - new_value._double;

  src_node->Grad1 = Grad1;
  src_node->Grad2 = Eigen::MatrixXd::Zero(2, 1);
  nmc->eval(det_nodes);
  nmc->compute_gradients(det_nodes);

  double new_sto_affected_nodes_log_prob;
  auto new_prop = create_proposer_dirichlet_beta(
      sto_nodes,
      tgt_node,
      param_a,
      param_b,
      new_value,
      /* out */ new_sto_affected_nodes_log_prob);
  double logacc = new_sto_affected_nodes_log_prob -
      old_sto_affected_nodes_log_prob + new_prop->log_prob(old_value) -
      old_prop->log_prob(new_value);
  // Accept or reject, reset (values and) gradients
  bool accepted = util::flip_coin_with_log_prob(nmc->gen, logacc);
  if (!accepted) {
    nmc->restore_old_values(det_nodes);
    *(src_node->value._matrix.data()) = old_X_k;
    *(src_node->value._matrix.data() + 1) = 1 - old_X_k;
  }
  // Gradients are must be cleared (equal to 0)
  // at the end of each iteration.
  // Some code relies on that to decide whether a node
  // is the one we are computing gradients with respect to.
  nmc->clear_gradients(det_nodes);
  tgt_node->grad1 = 0;
  tgt_node->grad2 = 0;
  graph->pd_finish(ProfilerEvent::NMC_STEP_DIRICHLET);
}

std::unique_ptr<proposer::Proposer>
NMCDirichletBetaSingleSiteStepper::create_proposer_dirichlet_beta(
    const std::vector<Node*>& sto_nodes,
    Node* tgt_node,
    double param_a,
    double param_b,
    NodeValue value,
    /* out */ double& logweight) {
  // TODO: Reorganize in the same manner the default NMC
  // proposer has been reorganized
  logweight = 0;
  double grad1 = 0;
  double grad2 = 0;
  for (Node* node : sto_nodes) {
    if (node == tgt_node) {
      double x = value._double;
      // X_k ~ Beta(param_a, param_b)
      logweight += (param_a - 1) * log(x) + (param_b - 1) * log(1 - x) +
          lgamma(param_a + param_b) - lgamma(param_a) - lgamma(param_b);

      grad1 += (param_a - 1) / x - (param_b - 1) / (1 - x);
      grad2 += -(param_a - 1) / (x * x) - (param_b - 1) / ((1 - x) * (1 - x));
    } else {
      logweight += node->log_prob();
      node->gradient_log_prob(tgt_node, /* in-out */ grad1, /* in-out */ grad2);
    }
  }

  std::unique_ptr<proposer::Proposer> prop =
      proposer::nmc_proposer(value, grad1, grad2);
  return prop;
}

} // namespace graph
} // namespace beanmachine
