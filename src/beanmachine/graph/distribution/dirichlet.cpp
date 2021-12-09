/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include "beanmachine/graph/distribution/dirichlet.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace distribution {

Dirichlet::Dirichlet(
    graph::ValueType sample_type,
    const std::vector<graph::Node*>& in_nodes)
    : Distribution(graph::DistributionType::DIRICHLET, sample_type) {
  // a Dirichlet has one parent which is a positive real matrix
  // and outputs a col simplex matrix
  if (sample_type.atomic_type != graph::AtomicType::PROBABILITY) {
    throw std::invalid_argument("Dirichlet produces probability samples");
  }
  if (sample_type.variable_type != graph::VariableType::COL_SIMPLEX_MATRIX) {
    throw std::invalid_argument(
        "Dirichlet produces COL_SIMPLEX_MATRIX samples");
  }
  if (in_nodes.size() != 1) {
    throw std::invalid_argument(
        "Dirichlet distribution must have exactly one parent");
  }
  if (in_nodes[0]->value.type.variable_type !=
          graph::VariableType::BROADCAST_MATRIX or
      in_nodes[0]->value.type.atomic_type != graph::AtomicType::POS_REAL or
      in_nodes[0]->value.type.cols != 1) {
    throw std::invalid_argument(
        "Dirichlet parent must be a positive real-valued matrix with one column.");
  }
}

Eigen::MatrixXd Dirichlet::_matrix_sampler(std::mt19937& gen) const {
  int n_rows = static_cast<int>(in_nodes[0]->value._matrix.rows());
  Eigen::MatrixXd sample(n_rows, 1);

  Eigen::MatrixXd param = in_nodes[0]->value._matrix;
  for (int i = 0; i < n_rows; i++) {
    std::gamma_distribution<double> gamma_dist(param(i), 1);
    sample(i) = gamma_dist(gen);
  }
  return sample / sample.sum();
}

double Dirichlet::log_prob(const graph::NodeValue& value) const {
  assert(value.type.variable_type == graph::VariableType::COL_SIMPLEX_MATRIX);
  assert(value.type.cols == 1);
  Eigen::MatrixXd param = in_nodes[0]->value._matrix;

  double log_prob = 0.0;
  for (int i = 0; i < param.size(); i++) {
    double alpha = param(i);
    log_prob -= lgamma(alpha);
    log_prob += std::log(value._matrix(i)) * (alpha - 1);
  }
  log_prob += lgamma(param.sum());

  return log_prob;
}

void Dirichlet::log_prob_iid(
    const graph::NodeValue& /* value */,
    Eigen::MatrixXd& /* log_probs */) const {}

// log_prob(x | a) =
// log G(\sum(a_i)) - \sum(log G(a_i)) + \sum(log(x_i) * (a_i - 1))
// grad1 w.r.t. x_i = (a_i - 1) / x_i
// grad1 w.r.t. a_i = digamma(\sum(a_i)) - digamma(a_i) + log(x_i)
// First order chain rule: f(g(x))' = f'(g(x)) g'(x)
void Dirichlet::backward_value(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    double adjunct) const {
  assert(value.type.variable_type == graph::VariableType::COL_SIMPLEX_MATRIX);
  Eigen::MatrixXd x = value._matrix;
  Eigen::MatrixXd param = in_nodes[0]->value._matrix;
  for (int i = 0; i < param.size(); i++) {
    back_grad._matrix(i) += adjunct * (param(i) - 1) / x(i);
  }
}

void Dirichlet::backward_param(const graph::NodeValue& value, double adjunct)
    const {
  assert(value.type.variable_type == graph::VariableType::COL_SIMPLEX_MATRIX);
  Eigen::MatrixXd x = value._matrix;
  Eigen::MatrixXd param = in_nodes[0]->value._matrix;
  double digamma_sum = util::polygamma(0, param.sum());
  if (in_nodes[0]->needs_gradient()) {
    for (int i = 0; i < param.size(); i++) {
      double jacob =
          std::log(x(i)) + digamma_sum - util::polygamma(0, param(i));
      in_nodes[0]->back_grad1._matrix(i) += adjunct * jacob;
    }
  }
}

} // namespace distribution
} // namespace beanmachine
