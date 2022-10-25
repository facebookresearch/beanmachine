/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define _USE_MATH_DEFINES
#include <cmath>

#include "beanmachine/graph/distribution/lkj_cholesky.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace distribution {

using namespace graph;

/* References:
 * https://pytorch.org/docs/stable/_modules/torch/distributions/lkj_cholesky.html
 * `Generating random correlation matrices based on vines and extended
 * onion method`, by Daniel Lewandowski, Dorota Kurowicka, Harry Joe.
 */

LKJCholesky::LKJCholesky(
    graph::ValueType sample_type,
    const std::vector<Node*>& in_nodes)
    : Distribution(DistributionType::LKJ_CHOLESKY, sample_type, in_nodes) {
  // an LKJ distribution has one parent, the concentration/shape parameter of
  // the distribution
  if (in_nodes.size() != 1) {
    throw std::invalid_argument(
        "LKJ Cholesky distribution must have one parent");
  }

  if (in_nodes[0]->value.type != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "LKJ Cholesky's parent must be a positive real number");
  }

  if (sample_type.variable_type != graph::VariableType::BROADCAST_MATRIX) {
    throw std::invalid_argument(
        "LKJ Cholesky produces BROADCAST_MATRIX samples");
  }

  if (sample_type.atomic_type != graph::AtomicType::REAL) {
    throw std::invalid_argument("LKJ Cholesky produces real samples");
  }

  d = sample_type.rows;
  if (d != sample_type.cols || d < 2) {
    throw std::invalid_argument(
        "LKJ Cholesky should produce a square matrix of dimension >= 2");
  }

  // Construct a vector [0, 0, 1, ..., d-2] + 0.5
  beta_conc1 = Eigen::VectorXd::LinSpaced(d, -0.5, d - 1.5);
  beta_conc1(0) = 0.5;
}

Eigen::VectorXd LKJCholesky::beta_conc0() const {
  double marginal_conc = in_nodes[0]->value._double + 0.5 * (d - 2);

  // Construct a vector marginal_conc - 0.5 * [0, 0, 1, ..., d-2]
  Eigen::VectorXd result = Eigen::VectorXd::LinSpaced(
      d, marginal_conc + 0.5, marginal_conc - 0.5 * d + 1);
  result(0) = marginal_conc;
  return result;
}

Eigen::ArrayXd LKJCholesky::order() const {
  // This is an intermediate field used by log_prob and its gradients.
  return Eigen::VectorXd(
             2.0 * (in_nodes[0]->value._double - 1) + (double)d -
             Eigen::ArrayXf::LinSpaced(d - 1, 2.0, (float)d).cast<double>())
      .array();
}

Eigen::MatrixXd LKJCholesky::_matrix_sampler(std::mt19937& gen) const {
  Eigen::VectorXd beta_result(d);
  Eigen::MatrixXd normal_result(d, d);

  auto beta_c0 = beta_conc0();
  for (uint i = 0; i < d; i++) {
    beta_result(i) = util::sample_beta(gen, beta_conc1(i), beta_c0(i));
  }

  // Sample a lower-triangular (excluding diagonal) matrix of standard normal
  // values.
  std::normal_distribution<double> dist(0.0, 1.0);
  for (uint i = 1; i < d; i++) { // Skips first row, this will be zeroed later
    for (uint j = 0; j < d; j++) {
      if (j < i) {
        normal_result(i, j) = dist(gen);
      } else {
        normal_result(i, j) = 0.0;
      }
    }
  }

  normal_result.rowwise().normalize();

  // Zero first row after normalization, since there will be nans
  for (uint i = 0; i < d; i++) {
    normal_result(0, i) = 0.0;
  }

  auto beta_sqrt = beta_result.cwiseSqrt().rowwise().replicate(d).array();
  auto w = Eigen::MatrixXd((beta_sqrt * normal_result.array()));

  // Clamp values before sqrt() for numerical stability
  auto diag =
      (1 - (w.array() * w.array()).rowwise().sum()).cwiseMax(1e-38).sqrt();

  // Set diagonal elements of w
  for (uint i = 0; i < d; i++) {
    w(i, i) = diag(i, 0);
  }

  return w;
}

double LKJCholesky::log_prob(const graph::NodeValue& value) const {
  uint dm1 = d - 1;
  auto diag_elems =
      value._matrix.diagonal().array()(Eigen::seq(1, Eigen::last));
  double eta = in_nodes[0]->value._double;
  auto unnormalized_log_pdf = (order() * diag_elems.log()).sum();

  // Compute normalization constant

  // log(denominator) = lgamma(alpha) * (d - 1)
  double log_pi = log(M_PI);
  double alpha = eta + 0.5 * dm1;
  double log_denom = lgamma(alpha) * dm1;

  // log(numerator) = multivariate_lgamma(alpha - 0.5, d-1)
  double log_numerator = log_pi * dm1 * (dm1 - 1.0) / 4.0;
  for (uint i = 1; i < d; i++) {
    log_numerator += lgamma(alpha - 0.5 - (i - 1.0) / 2.0);
  }

  double pi_constant = 0.5 * dm1 * log_pi;
  double log_norm_factor = pi_constant + log_numerator - log_denom;

  return unnormalized_log_pdf - log_norm_factor;
}

void LKJCholesky::gradient_log_prob_value(
    const graph::NodeValue& value,
    double& grad1,
    double& grad2) const {
  // The log_norm_factor from the log_prob computation is constant with respect
  // to value, so this is just the derivative of the unnormalized_log_pdf part.
  auto diag_elems =
      value._matrix.diagonal().array()(Eigen::seq(1, Eigen::last));
  auto o = order();
  grad1 += (o / diag_elems).sum();
  grad2 -= (o / (diag_elems * diag_elems)).sum();
}

void LKJCholesky::gradient_log_prob_param(
    const graph::NodeValue& value,
    double& grad1,
    double& grad2) const {
  uint dm1 = d - 1;
  double alpha = in_nodes[0]->value._double + 0.5 * dm1;

  // from contribution (through order) to unnormalized log pdf
  auto diag_elems =
      value._matrix.diagonal().array()(Eigen::seq(1, Eigen::last));
  grad1 += 2 * diag_elems.log().sum();

  // from normalization factor denominator
  grad1 += dm1 * util::polygamma(0, alpha);
  grad2 += dm1 * util::polygamma(1, alpha);

  // from normalization factor numerator
  for (uint i = 1; i < d; i++) {
    grad1 -= util::polygamma(0, alpha - i / 2.0);
    grad2 -= util::polygamma(1, alpha - i / 2.0);
  }
}

void LKJCholesky::backward_value(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    double adjunct) const {
  auto diag_elems =
      value._matrix.diagonal().array()(Eigen::seq(1, Eigen::last));
  auto o = order();
  auto grad_diagonal = adjunct * o / diag_elems;

  for (uint i = 1; i < d; i++) {
    back_grad(i, i) += grad_diagonal(i - 1);
  }
}

void LKJCholesky::backward_param(const graph::NodeValue& value, double adjunct)
    const {
  if (in_nodes[0]->needs_gradient()) {
    uint dm1 = d - 1;
    double alpha = in_nodes[0]->value._double + 0.5 * dm1;

    // from contribution (through order) to unnormalized log pdf
    auto diag_elems =
        value._matrix.diagonal().array()(Eigen::seq(1, Eigen::last));
    in_nodes[0]->back_grad1 += 2 * adjunct * diag_elems.log().sum();

    // from normalization factor denominator
    in_nodes[0]->back_grad1 += dm1 * adjunct * util::polygamma(0, alpha);

    // from normalization factor numerator
    for (uint i = 1; i < d; i++) {
      in_nodes[0]->back_grad1 -= adjunct * util::polygamma(0, alpha - i / 2.0);
    }
  }
};

} // namespace distribution
} // namespace beanmachine
