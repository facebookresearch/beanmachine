// Copyright (c) Facebook, Inc. and its affiliates.

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

ProfilerEvent NMCDirichletBetaSingleSiteStepper::get_step_profiler_event() {
  return ProfilerEvent::NMC_STEP_DIRICHLET;
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
NMCDirichletBetaSingleSiteStepper::get_proposal_distribution(Node* tgt_node) {
  assert(tgt_node->value._matrix.size() == 2);

  auto sto_tgt_node = static_cast<oper::StochasticOperator*>(tgt_node);
  double x = sto_tgt_node->value._matrix.coeff(0);

  // Propagate gradients
  // Prepare gradients of Dirichlet values wrt Beta value.
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
  mh->compute_gradients(mh->get_det_affected_nodes(tgt_node));

  // Use gradients to obtain NMC proposal
  // @lint-ignore CLANGTIDY
  auto dirichlet_distribution = sto_tgt_node->in_nodes[0];
  auto dirichlet_parameters_node = dirichlet_distribution->in_nodes[0];
  auto dirichlet_parameters_matrix = dirichlet_parameters_node->value._matrix;
  auto param_a = dirichlet_parameters_matrix.coeff(0);
  auto param_b = dirichlet_parameters_matrix.coeff(1);

  double grad1 = 0;
  double grad2 = 0;

  for (Node* node : mh->get_sto_affected_nodes(tgt_node)) {
    if (node == tgt_node) {
      // X ~ Beta(param_a, param_b)
      grad1 += (param_a - 1) / x - (param_b - 1) / (1 - x);
      grad2 += -(param_a - 1) / (x * x) - (param_b - 1) / ((1 - x) * (1 - x));
    } else {
      node->gradient_log_prob(tgt_node, /* in-out */ grad1, /* in-out */ grad2);
    }
  }

  // Obtain Beta proposal
  auto beta_sample_node_value = NodeValue(AtomicType::PROBABILITY, x);
  auto beta_proposal =
      proposer::nmc_proposer(beta_sample_node_value, grad1, grad2);

  // Wrap Beta proposal within Dirichlet proposal
  auto dirichlet_proposal =
      std::make_unique<proposer::FromProbabilityToDirichletProposerAdapter>(
          std::move(beta_proposal));

  return dirichlet_proposal;
}

} // namespace graph
} // namespace beanmachine
