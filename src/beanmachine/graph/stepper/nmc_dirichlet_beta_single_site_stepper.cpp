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
    const std::vector<Node*>& det_nodes,
    const std::vector<Node*>& sto_nodes) {
  graph->pd_begin(ProfilerEvent::NMC_STEP_DIRICHLET);
  assert(tgt_node->value._matrix.size() == 2);
  auto src_node = static_cast<oper::StochasticOperator*>(tgt_node);
  // @lint-ignore CLANGTIDY
  // in_node[0] is the Dirichlet distribution from which this node is sampled.
  // in_node[0]
  auto dirichlet_distribution = src_node->in_nodes[0];
  auto dirichlet_parameters_node = dirichlet_distribution->in_nodes[0];
  auto dirichlet_parameters_matrix = dirichlet_parameters_node->value._matrix;
  auto param_a = dirichlet_parameters_matrix.coeff(0);
  auto param_b = dirichlet_parameters_matrix.coeff(1);

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
  src_node->Grad1 = Grad1;
  src_node->Grad2 = Eigen::MatrixXd::Zero(2, 1);

  // Propagate gradients
  auto old_X = src_node->value._matrix.coeff(0);
  NodeValue old_value(AtomicType::PROBABILITY, old_X);
  nmc->save_old_values(det_nodes);
  nmc->compute_gradients(det_nodes);
  double old_sto_affected_nodes_log_prob = nmc->compute_log_prob_of(sto_nodes);
  auto old_prop = create_proposer_dirichlet_beta(
      sto_nodes, tgt_node, param_a, param_b, old_value);

  // We sample a new value for X and update Y as a result.
  NodeValue new_value = nmc->sample(old_prop);
  *(src_node->value._matrix.data()) = new_value._double;
  *(src_node->value._matrix.data() + 1) = 1 - new_value._double;

  // In general we would have to update Grad_i given the new X,
  // but since they are constant functions, they remain the same as before.
  // Showing it here for completeness:
  // src_node->Grad1 = Grad1;
  // src_node->Grad2 = Eigen::MatrixXd::Zero(2, 1);
  nmc->eval(det_nodes);
  nmc->compute_gradients(det_nodes);

  double new_sto_affected_nodes_log_prob = nmc->compute_log_prob_of(sto_nodes);
  auto new_prop = create_proposer_dirichlet_beta(
      sto_nodes, tgt_node, param_a, param_b, new_value);
  double logacc = new_sto_affected_nodes_log_prob -
      old_sto_affected_nodes_log_prob + new_prop->log_prob(old_value) -
      old_prop->log_prob(new_value);
  // Accept or reject, reset (values and) gradients
  bool accepted = util::flip_coin_with_log_prob(nmc->gen, logacc);
  if (!accepted) {
    nmc->restore_old_values(det_nodes);
    *(src_node->value._matrix.data()) = old_X;
    *(src_node->value._matrix.data() + 1) = 1 - old_X;
  }
  // Gradients are must be cleared (equal to 0)
  // at the end of each iteration.
  // Some code relies on that to decide whether a node
  // is the one we are computing gradients with respect to.
  nmc->clear_gradients(det_nodes);
  graph->pd_finish(ProfilerEvent::NMC_STEP_DIRICHLET);
}

std::unique_ptr<proposer::Proposer>
NMCDirichletBetaSingleSiteStepper::create_proposer_dirichlet_beta(
    const std::vector<Node*>& sto_nodes,
    Node* tgt_node,
    double param_a,
    double param_b,
    NodeValue value) {
  // TODO: Reorganize in the same manner the default NMC
  // proposer has been reorganized
  double grad1 = 0;
  double grad2 = 0;
  for (Node* node : sto_nodes) {
    if (node == tgt_node) {
      double x = value._double;
      // X ~ Beta(param_a, param_b)
      grad1 += (param_a - 1) / x - (param_b - 1) / (1 - x);
      grad2 += -(param_a - 1) / (x * x) - (param_b - 1) / ((1 - x) * (1 - x));
    } else {
      node->gradient_log_prob(tgt_node, /* in-out */ grad1, /* in-out */ grad2);
    }
  }

  std::unique_ptr<proposer::Proposer> prop =
      proposer::nmc_proposer(value, grad1, grad2);
  return prop;
}

} // namespace graph
} // namespace beanmachine
