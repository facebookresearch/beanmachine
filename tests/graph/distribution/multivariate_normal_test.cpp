/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <stdexcept>

#include "beanmachine/graph/distribution/multivariate_normal.h"
#include "beanmachine/graph/graph.h"

using namespace beanmachine::graph;

TEST(testdistrib, multivariate_normal_negative) {
  Graph g;

  Eigen::MatrixXd m1(3, 1);
  m1 << 1.5, 1.0, 2.0;
  Eigen::MatrixXd m2(3, 3);
  m2 << 3.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0; // not positive definite
  Eigen::MatrixXd m3(3, 3);
  m3 << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;

  uint cm1 = g.add_constant_real_matrix(m1);
  uint cm2 = g.add_constant_real_matrix(m2);
  uint cm3 = g.add_constant_real_matrix(m3);

  // negative initialization tests

  // invalid number of parents
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::MULTIVARIATE_NORMAL,
          ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 1),
          std::vector<uint>{}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::MULTIVARIATE_NORMAL,
          ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 1),
          std::vector<uint>{cm1}),
      std::invalid_argument);
  // invalid sample type
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::MULTIVARIATE_NORMAL,
          ValueType(
              VariableType::COL_SIMPLEX_MATRIX, AtomicType::PROBABILITY, 3, 1),
          std::vector<uint>{cm1, cm3}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::MULTIVARIATE_NORMAL,
          ValueType(VariableType::BROADCAST_MATRIX, AtomicType::NEG_REAL, 3, 1),
          std::vector<uint>{cm1, cm3}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::MULTIVARIATE_NORMAL,
          ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 2, 1),
          std::vector<uint>{cm1, cm3}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::MULTIVARIATE_NORMAL,
          ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 2),
          std::vector<uint>{cm1, cm3}),
      std::invalid_argument);
  auto llt = m2.llt();
  Eigen::MatrixXd temp = llt.matrixL();
  // invalid covariance matrix (not positive definite)
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::MULTIVARIATE_NORMAL,
          ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 1),
          std::vector<uint>{cm1, cm2}),
      std::invalid_argument);
}

TEST(testdistrib, multivariate_normal) {
  Graph g;
  Eigen::MatrixXd m1(3, 1);
  m1 << 1.5, 1.0, 2.0;

  Eigen::MatrixXd m2(3, 3);
  m2 << 1.0, 0.5, 0.0, 0.5, 1.0, 0.25, 0.0, 0.25, 1.0;

  uint flat_dist_mean = g.add_distribution(
      DistributionType::FLAT,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 1),
      std::vector<uint>{});
  uint flat_sample_mean =
      g.add_operator(OperatorType::SAMPLE, {flat_dist_mean});

  uint flat_dist_cov = g.add_distribution(
      DistributionType::FLAT,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 3),
      std::vector<uint>{});
  uint flat_sample_cov = g.add_operator(OperatorType::SAMPLE, {flat_dist_cov});

  g.observe(flat_sample_mean, m1);
  g.observe(flat_sample_cov, m2);

  uint multivariate_dist = g.add_distribution(
      DistributionType::MULTIVARIATE_NORMAL,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 1),
      std::vector<uint>{flat_sample_mean, flat_sample_cov});

  // test log prob
  uint mv_sample = g.add_operator(OperatorType::SAMPLE, {multivariate_dist});
  Eigen::MatrixXd obs(3, 1);
  obs << 1.0, 0.8, 1.5;
  g.observe(mv_sample, obs);
  EXPECT_NEAR(g.full_log_prob(), -2.8417, 0.01);

  // test backward_param() and backward_value()
  // verify with PyTorch
  // mean = torch.tensor([1.5, 1., 2.], requires_grad=True)
  // cov = torch.tensor([[1.0, 0.5, 0.0], [0.5, 1.0, 0.25], [0.0, 0.25, 1.0]],
  // requires_grad=True) mv = torch.distributions.MultivariateNormal(mean, cov)
  // value = torch.tensor([1.0, 0.8, 1.5], requires_grad=True)
  // log_prob = mv.log_prob(value)
  // torch.autograd.grad(log_prob, mean, retain_graph=True)
  // torch.autograd.grad(log_prob, cov, retain_graph=True)
  // torch.autograd.grad(log_prob, value)

  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 3);
  EXPECT_NEAR((*grad1[0])(0), -0.6273, 1e-3); // mean
  EXPECT_NEAR((*grad1[0])(1), 0.2545, 1e-3);
  EXPECT_NEAR((*grad1[0])(2), -0.5636, 1e-3);
  EXPECT_NEAR((*grad1[1])(0), -0.4851, 1e-3); // cov
  EXPECT_NEAR((*grad1[1])(1), 0.2838, 1e-3);
  EXPECT_NEAR((*grad1[1])(2), 0.0859, 1e-3);
  EXPECT_NEAR((*grad1[1])(3), 0.2838, 1e-3);
  EXPECT_NEAR((*grad1[1])(4), -0.6949, 1e-3);
  EXPECT_NEAR((*grad1[1])(5), 0.1101, 1e-3);
  EXPECT_NEAR((*grad1[1])(6), 0.0859, 1e-3);
  EXPECT_NEAR((*grad1[1])(7), 0.1101, 1e-3);
  EXPECT_NEAR((*grad1[1])(8), -0.3866, 1e-3);
  EXPECT_NEAR((*grad1[2])(0), 0.6273, 1e-3); // value
  EXPECT_NEAR((*grad1[2])(1), -0.2545, 1e-3);
  EXPECT_NEAR((*grad1[2])(2), 0.5636, 1e-3);
}

TEST(testdistrib, multivariate_normal_sample) {
  Graph g;
  Eigen::MatrixXd m1(3, 1);
  m1 << 1.5, 1.0, 2.0;

  Eigen::MatrixXd m2(3, 3);
  m2 << 1.0, 0.5, 0.0, 0.5, 1.0, 0.25, 0.0, 0.25, 1.0;

  uint cm1 = g.add_constant_real_matrix(m1);
  uint cm2 = g.add_constant_real_matrix(m2);
  uint mv_dist = g.add_distribution(
      DistributionType::MULTIVARIATE_NORMAL,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 1),
      std::vector<uint>{cm1, cm2});
  // test mean and covariance statistics
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{mv_dist});
  g.query(x);
  std::vector<std::vector<NodeValue>> samples =
      g.infer(100000, InferenceType::REJECTION);
  Eigen::MatrixXd mean(3, 1);
  Eigen::MatrixXd cov(3, 3);
  for (int i = 0; i < samples.size(); i++) {
    mean += samples[i][0]._matrix;
  }
  mean /= samples.size();

  for (int i = 0; i < samples.size(); i++) {
    Eigen::MatrixXd sample = samples[i][0]._matrix;
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        cov(j, k) += (sample(j) - mean(j)) * (sample(k) - mean(k));
      }
    }
  }
  cov /= samples.size() - 1;
  EXPECT_NEAR(mean(0), 1.5, 0.01);
  EXPECT_NEAR(mean(1), 1.0, 0.01);
  EXPECT_NEAR(mean(2), 2.0, 0.01);
  EXPECT_NEAR(cov(0, 0), 1.0, 0.01);
  EXPECT_NEAR(cov(0, 1), 0.5, 0.01);
  EXPECT_NEAR(cov(0, 2), 0.0, 0.01);
  EXPECT_NEAR(cov(1, 0), 0.5, 0.01);
  EXPECT_NEAR(cov(1, 1), 1.0, 0.01);
  EXPECT_NEAR(cov(1, 2), 0.25, 0.01);
  EXPECT_NEAR(cov(2, 0), 0.0, 0.01);
  EXPECT_NEAR(cov(2, 1), 0.25, 0.01);
  EXPECT_NEAR(cov(2, 2), 1.0, 0.01);
}

TEST(testdistrib, lkj_multivariate_normal_sample) {
  Graph g;
  Eigen::MatrixXd m1(3, 1);
  m1 << 1.5, 1.0, 2.0;

  uint cm1 = g.add_constant_real_matrix(m1);
  uint eta = g.add_constant_pos_real(3.0);
  uint lkj_chol_dist = g.add_distribution(
      DistributionType::LKJ_CHOLESKY,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 3),
      std::vector<uint>{eta});
  uint cov_llt =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{lkj_chol_dist});
  uint cov_llt_t =
      g.add_operator(OperatorType::TRANSPOSE, std::vector<uint>{cov_llt});
  uint cov = g.add_operator(
      OperatorType::MATRIX_MULTIPLY, std::vector<uint>{cov_llt, cov_llt_t});
  uint mv_dist = g.add_distribution(
      DistributionType::MULTIVARIATE_NORMAL,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 1),
      std::vector<uint>{cm1, cov});

  // test mean
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{mv_dist});
  g.query(x);
  std::vector<std::vector<NodeValue>> samples =
      g.infer(100000, InferenceType::REJECTION);
  Eigen::MatrixXd mean(3, 1);
  for (int i = 0; i < samples.size(); i++) {
    mean += samples[i][0]._matrix;
  };
  mean /= samples.size();

  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(mean(i), m1(i), 0.01);
  }
}
