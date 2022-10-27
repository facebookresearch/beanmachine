/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define _USE_MATH_DEFINES
#include <cmath>

#include <beanmachine/graph/distribution/multivariate_normal.h>
#include <beanmachine/graph/graph.h>

namespace beanmachine {
namespace distribution {
MultivariateNormal::MultivariateNormal(
    graph::ValueType sample_type,
    const std::vector<graph::Node*>& in_nodes)
    : Distribution(
          graph::DistributionType::MULTIVARIATE_NORMAL,
          sample_type,
          in_nodes) {
  // a multivariate normal has two parents which are a mean vector and a
  // covariance matrix
  // it outputs a (col) broadcast matrix
  if (sample_type.atomic_type != graph::AtomicType::REAL) {
    throw std::invalid_argument("Multivariate Normal produces real samples");
  }
  if (sample_type.variable_type != graph::VariableType::BROADCAST_MATRIX) {
    throw std::invalid_argument(
        "Multivariate Normal produces BROADCAST_MATRIX samples");
  }
  if (in_nodes.size() != 2) {
    throw std::invalid_argument(
        "Multivariate Normal must have exactly two parents");
  }
  if (in_nodes[0]->value.type.variable_type !=
          graph::VariableType::BROADCAST_MATRIX or
      in_nodes[0]->value.type.atomic_type != graph::AtomicType::REAL or
      in_nodes[0]->value.type.cols != 1) {
    throw std::invalid_argument(
        "Multivariate Normal's first parent must be a real-valued matrix with one column");
  }
  uint rows = in_nodes[0]->value.type.rows;
  if (in_nodes[1]->value.type.variable_type !=
          graph::VariableType::BROADCAST_MATRIX or
      in_nodes[1]->value.type.atomic_type != graph::AtomicType::REAL or
      in_nodes[1]->value.type.rows != rows or
      in_nodes[1]->value.type.cols != rows) {
    throw std::invalid_argument(
        "Multivariate Normal's second parent must be a real-valued square matrix with the same number of rows as the first parent");
  }
  if (sample_type.rows != rows or sample_type.cols != 1) {
    throw std::invalid_argument(
        "Multivariate Normal's sample type should match the shape of the first parent");
  }
  // We also require that the covariance matrix be positive definite. We check
  // this using Cholesky decomposition and store the lower triangular matrix for
  // use in sampling later.
  if (in_nodes[1]->node_type == graph::NodeType::CONSTANT) {
    // LLT is the Eigen operation for Cholesky decomposition. We store this
    // value for constant covariance to avoid recomputation.

    _llt = in_nodes[1]->value._matrix.llt();
    if (_llt.info() == Eigen::NumericalIssue) {
      throw std::invalid_argument(
          "Multivariate Normal's covariance matrix must be positive definite");
    }
  }
}

Eigen::LLT<Eigen::MatrixXd> MultivariateNormal::llt() const {
  if (in_nodes[1]->node_type == graph::NodeType::CONSTANT) {
    return _llt;
  } else {
    // If the covariance is not constant, we need to recompute the
    // Cholesky decomposition each time we sample or take the log prob.
    auto result = in_nodes[1]->value._matrix.llt();
    if (result.info() == Eigen::NumericalIssue) {
      throw std::invalid_argument(
          "Multivariate Normal's covariance matrix must be positive definite");
    }
    return result;
  }
}

Eigen::MatrixXd MultivariateNormal::_matrix_sampler(std::mt19937& gen) const {
  // We generate samples of a multivariate normal through an affine transform of
  // N independent standard normal samples x = u + Az where A is a real matrix
  // such that A * A^T = \Sigma, typically calculated using the Cholesky
  // decomposition.
  int n_rows = static_cast<int>(in_nodes[0]->value._matrix.rows());
  Eigen::MatrixXd sample(n_rows, 1);
  std::normal_distribution<double> standard_normal(0, 1);

  for (int i = 0; i < n_rows; i++) {
    sample(i) = standard_normal(gen);
  }

  Eigen::MatrixXd mean = in_nodes[0]->value._matrix;
  return llt().matrixL() * sample + mean;
}

double MultivariateNormal::log_prob(const graph::NodeValue& value) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  assert(value.type.cols == 1);

  Eigen::MatrixXd x = value._matrix;
  Eigen::MatrixXd mean = in_nodes[0]->value._matrix;
  int dims = static_cast<int>(in_nodes[0]->value._matrix.rows());

  auto computed_llt = llt();
  double mdist = computed_llt.matrixL().solve(x - mean).squaredNorm();
  const double log2pi = std::log(2 * M_PI);
  double log_det =
      2 * std::log(computed_llt.matrixL().nestedExpression().diagonal().prod());

  return -0.5 * (dims * log2pi + log_det + mdist);
}

// log_prob(x | u, S) =
// -1/2 (k*log(2\pi) + log|S| + (x - u)^T S^-1 (x - u))
// grad1 w.r.t x = -S^-1 (x - u)
// grad1 w.r.t u = S^-1 (x - u)
// grad1 w.r.t S = -1/2 (S^-1 - S^-1 (x - u)(x-u)^T S^-1)
void MultivariateNormal::backward_value(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    double adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  Eigen::MatrixXd x = value._matrix;
  Eigen::MatrixXd mean = in_nodes[0]->value._matrix;
  Eigen::MatrixXd sigma = in_nodes[1]->value._matrix;
  back_grad += adjunct * -sigma.inverse() * (x - mean);
}
void MultivariateNormal::backward_param(
    const graph::NodeValue& value,
    double adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  Eigen::MatrixXd x = value._matrix;
  Eigen::MatrixXd mean = in_nodes[0]->value._matrix;
  Eigen::MatrixXd sigma = in_nodes[1]->value._matrix;
  Eigen::MatrixXd sigma_inv = sigma.inverse();
  // Because the above values are of type Eigen::MatrixXd
  // the * operation is a matrix multiply.
  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1 += adjunct * sigma_inv * (x - mean);
  }
  if (in_nodes[1]->needs_gradient()) {
    Eigen::MatrixXd err = x - mean;
    in_nodes[1]->back_grad1 += adjunct * (-0.5) *
        (sigma_inv - sigma_inv * err * err.transpose() * sigma_inv);
  }
}

} // namespace distribution
} // namespace beanmachine
