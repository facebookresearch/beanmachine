/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define _USE_MATH_DEFINES
#include <cmath>

#include <gtest/gtest.h>
#include <Eigen/Core>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/distribution/lkj_cholesky.h"
#include "beanmachine/graph/graph.h"

using namespace beanmachine::graph;

TEST(testdistrib, lkj_cholesky) {
  Graph g;
  const double ETA = 3.0;
  auto pos1 = g.add_constant_pos_real(ETA);
  auto nat1 = g.add_constant_natural(1);

  // negative tests that LKJ Cholesky has one positive real parent
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::LKJ_CHOLESKY,
          ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 5, 5),
          std::vector<uint>{}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::LKJ_CHOLESKY,
          ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 5, 5),
          std::vector<uint>{nat1}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::LKJ_CHOLESKY,
          ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 5, 5),
          std::vector<uint>{pos1, pos1}),
      std::invalid_argument);

  // negative tests for sample type
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::LKJ_CHOLESKY,
          AtomicType::REAL,
          std::vector<uint>{pos1}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::LKJ_CHOLESKY,
          ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 5),
          std::vector<uint>{pos1}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::LKJ_CHOLESKY,
          ValueType(VariableType::BROADCAST_MATRIX, AtomicType::BOOLEAN, 5, 5),
          std::vector<uint>{pos1}),
      std::invalid_argument);

  // test creation of a distribution
  auto lkj_chol_dist = g.add_distribution(
      DistributionType::LKJ_CHOLESKY,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 5, 5),
      std::vector<uint>{pos1});

  uint sample = g.add_operator(OperatorType::SAMPLE, {lkj_chol_dist});
}

TEST(testdistrib, lkj_cholesky_betas) {
  // One step in the onion method for sampling correlation matrices involves
  // creating a (vector) beta distribution. This checks that the concentrations
  // for the beta distribution match those generated by PyTorch for the same
  // values of eta and d in this implementation:
  //
  // https://pytorch.org/docs/stable/_modules/torch/distributions/lkj_cholesky.html
  //
  // See `Generating random correlation matrices based on vines and extended
  // onion method`, by Daniel Lewandowski, Dorota Kurowicka, Harry Joe.

  Graph g;
  const double ETA = 3.0;
  auto pos1 = g.add_constant_pos_real(ETA);
  auto parents = g.convert_node_ids(std::vector<uint>{pos1});

  auto lkj_chol_dist = std::make_unique<beanmachine::distribution::LKJCholesky>(
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 5, 5),
      parents);

  Eigen::VectorXd expected_b0(5), expected_b1(5);
  expected_b0 << 4.5, 4.5, 4.0, 3.5, 3.0;
  expected_b1 << 0.5, 0.5, 1.5, 2.5, 3.5;

  auto beta_conc0 = lkj_chol_dist->beta_conc0();
  for (int i = 0; i < 5; i++) {
    EXPECT_NEAR(beta_conc0(i), expected_b0(i), 1e-5);
    EXPECT_NEAR(lkj_chol_dist->beta_conc1(i), expected_b1(i), 1e-5);
  }
}

TEST(testdistrib, lkj_cholesky_sample) {
  Graph g;
  std::mt19937 mt1(0);
  const double ETA = 3.0;
  auto pos1 = g.add_constant_pos_real(ETA);
  auto parents = g.convert_node_ids(std::vector<uint>{pos1});

  auto lkj_chol_dist = std::make_unique<beanmachine::distribution::LKJCholesky>(
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 5, 5),
      parents);

  auto lkj_chol_sample = lkj_chol_dist->_matrix_sampler(mt1);
  auto lkj = lkj_chol_sample * lkj_chol_sample.transpose();

  // Check elements for these criteria:
  //    - Diagonals are 1s (each index is perfectly correlated with itself)
  //    - Other correlations are between -1 and 1
  //    - Matrix is symmetric
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j <= i; j++) {
      if (i == j) {
        EXPECT_NEAR(lkj(i, j), 1.0, 1e-5);
      } else {
        EXPECT_NEAR(lkj(i, j), lkj(j, i), 1e-5);
        EXPECT_GE(lkj(i, j), -1.0);
        EXPECT_LE(lkj(j, i), 1.0);
      }
    }
  }
}

TEST(testdistrib, lkj_cholesky_log_prob) {
  Graph g;
  const double ETA = 3.0;
  auto pos1 = g.add_constant_pos_real(ETA);

  auto lkj_chol_dist = g.add_distribution(
      DistributionType::LKJ_CHOLESKY,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 3),
      std::vector<uint>{pos1});

  Eigen::MatrixXd obs(3, 3);
  obs << 1.0, 0.0, 0.0, 0.1818, 0.9833, 0.0, 0.2349, 0.4351, 0.8692;

  uint lkj_chol_sample = g.add_operator(OperatorType::SAMPLE, {lkj_chol_dist});
  g.observe(lkj_chol_sample, obs);
  EXPECT_NEAR(-0.6724, g.full_log_prob(), 0.001);
}

TEST(testdistrib, lkj_cholesky_log_prob_stochastic_parent) {
  Graph g;

  auto a = g.add_constant_pos_real(3.0);
  auto b = g.add_constant_pos_real(1.0);
  auto gamma_dist = g.add_distribution(
      DistributionType::GAMMA, AtomicType::POS_REAL, std::vector<uint>{a, b});
  auto eta =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{gamma_dist});
  auto lkj_chol_dist = g.add_distribution(
      DistributionType::LKJ_CHOLESKY,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 3),
      std::vector<uint>{eta});

  Eigen::MatrixXd obs(3, 3);
  obs << 1.0, 0.0, 0.0, 0.1818, 0.9833, 0.0, 0.2349, 0.4351, 0.8692;

  uint lkj_chol_sample = g.add_operator(OperatorType::SAMPLE, {lkj_chol_dist});
  g.observe(eta, 3.0);
  g.observe(lkj_chol_sample, obs);
  EXPECT_NEAR(-2.16855, g.full_log_prob(), 0.001);
}

TEST(testdistrib, lkj_cholesky_log_prob_forward_value) {
  Graph g;
  const double ETA = 3.0;
  auto pos1 = g.add_constant_pos_real(ETA);

  /***
   * For PyTorch verification:
   *
   * torch.manual_seed(0)
   * dist = torch.distributions.LKJCholesky(3, 3.0)
   * x = dist.sample()    # this is the value copied for this test
   * x.requires_grad = True
   *
   * log_p = dist.log_prob(x)
   * grad = torch.autograd.grad(log_p, x)[0].sum()
   *
   * We needed to sample directly from the distribution (rather than
   * pasting in values) because inexact values could cause this to
   * fail the support check.
   ***/

  auto lkj_chol_dist = g.add_distribution(
      DistributionType::LKJ_CHOLESKY,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 3),
      std::vector<uint>{pos1});

  Eigen::MatrixXd obs(3, 3);
  obs << 1.0, 0.0, 0.0, 0.1206, 0.9927, 0.0, 0.1033, 0.4061, 0.9080;

  uint lkj_chol_sample = g.add_operator(OperatorType::SAMPLE, {lkj_chol_dist});
  g.observe(lkj_chol_sample, obs);

  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(lkj_chol_sample, grad1, grad2);
  EXPECT_NEAR(grad1, 9.4421, 0.01);
  EXPECT_NEAR(grad2, -9.9254, 0.01);
}

TEST(testdistrib, lkj_cholesky_log_prob_forward_param) {
  Graph g;
  auto scale = g.add_constant_pos_real(3.0);

  auto half_normal = g.add_distribution(
      DistributionType::HALF_NORMAL,
      AtomicType::POS_REAL,
      std::vector<uint>{scale});
  uint eta = g.add_operator(OperatorType::SAMPLE, {half_normal});
  g.observe(eta, 3.0);

  auto lkj_chol_dist = g.add_distribution(
      DistributionType::LKJ_CHOLESKY,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 3),
      std::vector<uint>{eta});

  Eigen::MatrixXd obs(3, 3);
  obs << 1.0, 0.0, 0.0, 0.1206, 0.9927, 0.0, 0.1033, 0.4061, 0.9080;

  uint lkj_chol_sample = g.add_operator(OperatorType::SAMPLE, {lkj_chol_dist});

  g.observe(lkj_chol_sample, obs);

  double grad1 = 0;
  double grad2 = 0;

  /***
   * For PyTorch verification of first derivative:
   *
   * torch.manual_seed(0)
   * eta = torch.tensor([0.3], requires_grad=True)
   * dist = torch.distributions.LKJCholesky(3, eta)
   * x = dist.sample()
   * # x needed to be sampled to avoid support validation issues, but the value
   * # used in this test is:
   * # x = torch.tensor([[1.0, 0.0, 0.0], [0.1206, 0.9927, 0.0], [0.1033,
   * #      0.4061, 0.9080]])
   *
   * log_p = dist.log_prob(x).sum() +
   *          torch.distributions.HalfNormal(3.0).log_prob(eta).sum() grad =
   * torch.autograd.grad(log_p, eta)[0].sum()
   *
   *
   * For second derivative:
   * log_pdf = order*log(val_diagonal) - normalize_numerator + normalize_denom
   *      d2Order/dEta^2 = 0
   *      alpha = eta + 0.5 * (d-1) = 3.0 + 0.5 * 2 = 4;
   *      normalize_numerator = mvlgamma(alpha - 0.5, d-1)
   *          = C + /sum_{i=1}^{d-1} lgamma(alpha - i/2)
   *        ---> second derivative = /sum_{i=1}^{d-1} polygamma(1, alpha - i/2)
   *      normalize_denom = (d-1) * lgamma(alpha)
   *        ---> second derivative = (d-1) * polygamma(1, alpha)
   * d2Log_pdf/dEta^2 = (d-1) * polygamma(1, alpha) -
   *        /sum_{i=1}^{d-1} polygamma(1, alpha - i/2)
   *      = (3-1) * polygamma(1, 4.0) + /sum_{i=1}^{d-1} polygamma(1, 4.0 - i/2)
   *      = 2 * polygamma(1, 4.0) - polygamma(1, 3.5) - polygamma(1, 3.0)
   *      = -0.157646
   * Also include the half-normal term (-0.11111) for a total grad2 of -0.2688
   ***/

  g.gradient_log_prob(eta, grad1, grad2);
  EXPECT_NEAR(grad1, -0.0548, 0.001);
  EXPECT_NEAR(grad2, -0.2688, 0.001);
}

TEST(testdistrib, lkj_cholesky_log_prob_backward_value) {
  Graph g;
  const double ETA = 3.0;
  auto pos1 = g.add_constant_pos_real(ETA);

  /***
   * For PyTorch verification:
   *
   * torch.manual_seed(0)
   * dist = torch.distributions.LKJCholesky(3, 3.0)
   * x = dist.sample()
   * # value used in this test is:
   * # x = torch.tensor([[1.0, 0.0, 0.0], [0.1206, 0.9927, 0.0], [0.1033,
   * #      0.4061, 0.9080]])
   * x.requires_grad = True
   *
   * log_p = dist.log_prob(x)
   * grad = torch.autograd.grad(log_p, x)[0]
   *
   * We needed to sample directly from the distribution (rather than
   * pasting in values) because inexact values could cause this to
   * fail the support check.
   ***/

  auto lkj_chol_dist = g.add_distribution(
      DistributionType::LKJ_CHOLESKY,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 3),
      std::vector<uint>{pos1});

  Eigen::MatrixXd obs(3, 3), expected_grad(3, 3);
  obs << 1.0, 0.0, 0.0, 0.1206, 0.9927, 0.0, 0.1033, 0.4061, 0.9080;
  expected_grad << 0.0000, 0.0000, 0.0000, 0.0000, 5.0367, 0.0000, 0.0000,
      0.0000, 4.4054;

  uint lkj_chol_sample = g.add_operator(OperatorType::SAMPLE, {lkj_chol_dist});
  g.observe(lkj_chol_sample, obs);

  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 1);

  auto grad = grad1[0]->as_matrix();

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      EXPECT_NEAR(grad(i, j), expected_grad(i, j), 0.001);
    }
  }
}

TEST(testdistrib, lkj_cholesky_log_prob_backward_value_2) {
  Graph g;
  const double ETA = 3.0;
  Eigen::MatrixXd mu(3, 1);
  mu << 3.0, 2.0, -1.5;
  auto pos1 = g.add_constant_pos_real(ETA);
  auto means = g.add_constant_real_matrix(mu);

  /***
   * For PyTorch verification:
   * # x needed to be sampled to avoid support validation issues. The value
   * # used in this test is:
   * # x = torch.tensor([[1.0, 0.0, 0.0], [0.1206, 0.9927, 0.0], [0.1033,
   * #      0.4061, 0.9080]])
   *
   * torch.manual_seed(0)
   * eta = torch.tensor([3.0])
   * mu = torch.tensor([3.0, 2.0, -1.5], requires_grad=True)
   * dist = torch.distributions.LKJCholesky(3, eta)
   * x = dist.sample()[0]
   * x.requires_grad = True
   * cov = torch.matmul(x, x.t())
   * mv_normal = torch.distributions.MultivariateNormal(mu, cov)
   * obs = torch.tensor([3.0, 3.0, -1.0])
   * log_p = mv_normal.log_prob(obs) + dist.log_prob(x).sum()
   * grad = torch.autograd.grad(log_p, x)[0]
   * # grad = tensor([[-1.0000e+00, -7.7852e-03,  4.6564e-02],
   * #  [ 2.8756e-10,  5.0062e+00,  5.4763e-01],
   * #  [-6.1548e-09,  1.1110e-01,  3.3151e+00]])
   ***/

  auto lkj_chol_dist = g.add_distribution(
      DistributionType::LKJ_CHOLESKY,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 3),
      std::vector<uint>{pos1});

  uint cov_llt = g.add_operator(OperatorType::SAMPLE, {lkj_chol_dist});
  uint cov_llt_t = g.add_operator(OperatorType::TRANSPOSE, {cov_llt});
  uint cov =
      g.add_operator(OperatorType::MATRIX_MULTIPLY, {cov_llt, cov_llt_t});

  auto mv_dist = g.add_distribution(
      DistributionType::MULTIVARIATE_NORMAL,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 1),
      std::vector<uint>{means, cov});

  uint mv_sample = g.add_operator(OperatorType::SAMPLE, {mv_dist});

  Eigen::MatrixXd obs(3, 1), obs2(3, 3), expected_grad(3, 3);
  obs << 3.0, 3.0, -1.0;
  obs2 << 1.0, 0.0, 0.0, 0.1206, 0.9927, 0.0, 0.1033, 0.4061, 0.9080;
  expected_grad << -1.0000, -0.007852, 0.046564, 0.0000, 5.0062, 0.54763,
      0.0000, 0.1110, 3.3151;

  uint lkj_chol_sample = g.add_operator(OperatorType::SAMPLE, {lkj_chol_dist});
  g.observe(cov_llt, obs2);
  g.observe(mv_sample, obs);

  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 2);

  auto grad = grad1[0]->as_matrix();

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      EXPECT_NEAR(grad(i, j), expected_grad(i, j), 0.001);
    }
  }
}

TEST(testdistrib, lkj_cholesky_log_prob_backward_param) {
  Graph g;
  auto scale = g.add_constant_pos_real(3.0);

  auto half_normal = g.add_distribution(
      DistributionType::HALF_NORMAL,
      AtomicType::POS_REAL,
      std::vector<uint>{scale});
  uint eta = g.add_operator(OperatorType::SAMPLE, {half_normal});
  g.observe(eta, 3.0);

  auto lkj_chol_dist = g.add_distribution(
      DistributionType::LKJ_CHOLESKY,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 3),
      std::vector<uint>{eta});

  Eigen::MatrixXd obs(3, 3);
  obs << 1.0, 0.0, 0.0, 0.1206, 0.9927, 0.0, 0.1033, 0.4061, 0.9080;

  uint lkj_chol_sample = g.add_operator(OperatorType::SAMPLE, {lkj_chol_dist});

  g.observe(lkj_chol_sample, obs);

  /***
   * For PyTorch verification of first derivative:
   *
   * torch.manual_seed(0)
   * eta = torch.tensor([3.0], requires_grad=True)
   * dist = torch.distributions.LKJCholesky(3, eta)
   * x = dist.sample()
   *
   * log_p = dist.log_prob(x).sum() +
   *          torch.distributions.HalfNormal(3.0).log_prob(eta).sum() grad =
   * torch.autograd.grad(log_p, eta)[0]
   ***/
  std::vector<DoubleMatrix*> grads;
  g.eval_and_grad(grads);

  EXPECT_EQ(grads.size(), 2);
  EXPECT_NEAR(*grads[0], -0.0548, 0.01);
}
