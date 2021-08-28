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
#include "beanmachine/graph/proposer/from_probability_to_dirichlet_proposer_adapter.h"
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

void NMCDirichletBetaSingleSiteStepper::step(
    Node* tgt_node,
    const std::vector<Node*>& det_affected_nodes,
    const std::vector<Node*>& sto_affected_nodes) {
  graph->pd_begin(ProfilerEvent::NMC_STEP_DIRICHLET);

  auto proposal_distribution_given_old_value = get_proposal_distribution(
      tgt_node, det_affected_nodes, sto_affected_nodes);

  NodeValue new_value = nmc->sample(proposal_distribution_given_old_value);

  nmc->revertibly_set_and_propagate(
      tgt_node, new_value, det_affected_nodes, sto_affected_nodes);

  double new_sto_affected_nodes_log_prob =
      nmc->compute_log_prob_of(sto_affected_nodes);

  auto proposal_distribution_given_new_value = get_proposal_distribution(
      tgt_node, det_affected_nodes, sto_affected_nodes);

  NodeValue& old_value = nmc->get_old_value(tgt_node);
  double old_sto_affected_nodes_log_prob =
      nmc->get_old_sto_affected_nodes_log_prob();

  double logacc = new_sto_affected_nodes_log_prob -
      old_sto_affected_nodes_log_prob +
      proposal_distribution_given_new_value->log_prob(old_value) -
      proposal_distribution_given_old_value->log_prob(new_value);

  bool accepted = util::flip_coin_with_log_prob(nmc->gen, logacc);
  if (!accepted) {
    nmc->revert_set_and_propagate(tgt_node, det_affected_nodes);
  }

  // Gradients must be cleared (made equal to 0)
  // at the end of each iteration.
  // Some code relies on that to decide whether a node
  // is the one we are computing gradients with respect to.
  nmc->clear_gradients(det_affected_nodes);
  graph->pd_finish(ProfilerEvent::NMC_STEP_DIRICHLET);
}

/*
 * An adapter to go from a base proposer producing a probability p
 * to a new proposer that produces a Dirichlet sample (p, 1 - p).
 */
class FromProbabilityToDirichletProposerAdapter : public proposer::Proposer {
 public:
  FromProbabilityToDirichletProposerAdapter(
      std::unique_ptr<proposer::Proposer> probability_proposer)
      : probability_proposer(std::move(probability_proposer)) {}

  virtual graph::NodeValue sample(std::mt19937& gen) const {
    auto x = probability_proposer->sample(gen);
    ValueType value_type(
        VariableType::COL_SIMPLEX_MATRIX, AtomicType::PROBABILITY, 2, 1);
    Eigen::MatrixXd values(2, 1);
    values << x._double, 1 - x._double;
    NodeValue y(value_type, values);
    return y;
  }

  virtual double log_prob(graph::NodeValue& value) const {
    NodeValue x(AtomicType::PROBABILITY, value._matrix.coeff(0));
    return probability_proposer->log_prob(x);
  }

 private:
  std::unique_ptr<proposer::Proposer> probability_proposer;
};

std::unique_ptr<proposer::Proposer>
NMCDirichletBetaSingleSiteStepper::get_proposal_distribution(
    Node* tgt_node,
    const std::vector<Node*>& det_affected_nodes,
    const std::vector<Node*>& sto_affected_nodes) {
  // TODO: Reorganize in the same manner the default NMC
  // proposer has been reorganized

  assert(tgt_node->value._matrix.size() == 2);

  auto sto_tgt_node = static_cast<oper::StochasticOperator*>(tgt_node);
  double x = sto_tgt_node->value._matrix.coeff(0);

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

  // Use gradients to obtain NMC proposal
  // @lint-ignore CLANGTIDY
  auto dirichlet_distribution = sto_tgt_node->in_nodes[0];
  auto dirichlet_parameters_node = dirichlet_distribution->in_nodes[0];
  auto dirichlet_parameters_matrix = dirichlet_parameters_node->value._matrix;
  auto param_a = dirichlet_parameters_matrix.coeff(0);
  auto param_b = dirichlet_parameters_matrix.coeff(1);

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

  // Wrap x proposal within y proposal
  auto x_node_value = NodeValue(AtomicType::PROBABILITY, x);
  auto x_proposal = proposer::nmc_proposer(x_node_value, grad1, grad2);
  auto y_proposal =
      std::make_unique<proposer::FromProbabilityToDirichletProposerAdapter>(
          std::move(x_proposal));
  return y_proposal;
}

} // namespace graph
} // namespace beanmachine
