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
We base the 2-dimensional Dirichlet-distributed (Y_1, Y_2)
as a deterministic function of a Beta-distributed X:
(Y_1, Y_2) = (X, 1 - X)
*/
void NMCDirichletBetaSingleSiteStepper::step(
    Node* tgt_node,
    const std::vector<Node*>& det_affected_nodes,
    const std::vector<Node*>& sto_affected_nodes) {
  graph->pd_begin(ProfilerEvent::NMC_STEP_DIRICHLET);
  assert(tgt_node->value._matrix.size() == 2);
  auto sto_tgt_node = static_cast<oper::StochasticOperator*>(tgt_node);

  auto old_X = sto_tgt_node->value._matrix.coeff(0);
  NodeValue old_value(AtomicType::PROBABILITY, old_X);
  nmc->save_old_values(det_affected_nodes);
  double old_sto_affected_nodes_log_prob =
      nmc->compute_log_prob_of(sto_affected_nodes);
  auto old_prop = get_proposal_distribution(
      sto_tgt_node, det_affected_nodes, sto_affected_nodes);

  // We sample a new value for X and update Y as a result.
  NodeValue new_value = nmc->sample(old_prop);
  *(sto_tgt_node->value._matrix.data()) = new_value._double;
  *(sto_tgt_node->value._matrix.data() + 1) = 1 - new_value._double;

  // In general we would have to update Grad_i given the new X,
  // but since they are constant functions, they remain the same as before.
  // Showing it here for completeness:
  // sto_tgt_node->Grad1 = Grad1;
  // sto_tgt_node->Grad2 = Eigen::MatrixXd::Zero(2, 1);
  nmc->eval(det_affected_nodes);
  nmc->compute_gradients(det_affected_nodes);

  double new_sto_affected_nodes_log_prob =
      nmc->compute_log_prob_of(sto_affected_nodes);
  auto new_prop = get_proposal_distribution(
      sto_tgt_node, det_affected_nodes, sto_affected_nodes);
  double logacc = new_sto_affected_nodes_log_prob -
      old_sto_affected_nodes_log_prob + new_prop->log_prob(old_value) -
      old_prop->log_prob(new_value);
  bool accepted = util::flip_coin_with_log_prob(nmc->gen, logacc);
  if (!accepted) {
    nmc->restore_old_values(det_affected_nodes);
    *(sto_tgt_node->value._matrix.data()) = old_X;
    *(sto_tgt_node->value._matrix.data() + 1) = 1 - old_X;
  }
  // Gradients are must be cleared (equal to 0)
  // at the end of each iteration.
  // Some code relies on that to decide whether a node
  // is the one we are computing gradients with respect to.
  nmc->clear_gradients(det_affected_nodes);
  graph->pd_finish(ProfilerEvent::NMC_STEP_DIRICHLET);
}

std::unique_ptr<proposer::Proposer>
NMCDirichletBetaSingleSiteStepper::get_proposal_distribution(
    Node* tgt_node,
    const std::vector<Node*>& det_affected_nodes,
    const std::vector<Node*>& sto_affected_nodes) {
  // TODO: Reorganize in the same manner the default NMC
  // proposer has been reorganized

  auto sto_tgt_node = static_cast<oper::StochasticOperator*>(tgt_node);
  double x = sto_tgt_node->value._matrix.coeff(0);

  // @lint-ignore CLANGTIDY
  auto dirichlet_distribution = sto_tgt_node->in_nodes[0];
  auto dirichlet_parameters_node = dirichlet_distribution->in_nodes[0];
  auto dirichlet_parameters_matrix = dirichlet_parameters_node->value._matrix;
  auto param_a = dirichlet_parameters_matrix.coeff(0);
  auto param_b = dirichlet_parameters_matrix.coeff(1);


  // Propagate gradients
  // Prepare gradients of Y wrt X.
  // Those are used by descendants to compute the log prod gradient
  // with respect to X.
  // Grad1 = (dY_1/dX_1, dY_2/dX_1)
  // where dY_1/dX_1 = 1
  //       dY_2/dX_1 = -1
  // Grad2 = ( (d/dX_1)^2 Y_1, (d/dX_1)^2 Y_2 )
  // where (d/dX_1)^2 Y_1 = 0
  //       (d/dX_1)^2 Y_2 = 0
  Eigen::MatrixXd Grad1(2, 1);
  Grad1 << 1, -1;
  sto_tgt_node->Grad1 = Grad1;
  sto_tgt_node->Grad2 = Eigen::MatrixXd::Zero(2, 1);
  nmc->compute_gradients(det_affected_nodes);

  double grad1 = 0;
  double grad2 = 0;

  for (Node* node : sto_affected_nodes) {
    if (node == tgt_node) {
      // X ~ Beta(param_a, param_b)
      grad1 += (param_a - 1) / x - (param_b - 1) / (1 - x);
      grad2 += -(param_a - 1) / (x * x) - (param_b - 1) / ((1 - x) * (1 - x));
    } else {
      node->gradient_log_prob(tgt_node, /* in-out */ grad1, /* in-out */ grad2);
    }
  }

  auto x_node_value = NodeValue(AtomicType::PROBABILITY, x);
  std::unique_ptr<proposer::Proposer> prop =
      proposer::nmc_proposer(x_node_value, grad1, grad2);
  return prop;
}

} // namespace graph
} // namespace beanmachine
