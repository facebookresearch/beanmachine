/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <stdexcept>

#include "beanmachine/graph/distribution/dirichlet.h"
#include "beanmachine/graph/graph.h"

using namespace beanmachine::graph;

TEST(testdistrib, dirichlet_negative) {
  Graph g;

  Eigen::MatrixXd m1(3, 1);
  m1 << 1.5, 1.0, 2.0;
  uint cm1 = g.add_constant_pos_matrix(m1);
  uint two = g.add_constant((natural_t)2);

  // negative initialization tests
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::DIRICHLET,
          ValueType(
              VariableType::COL_SIMPLEX_MATRIX, AtomicType::PROBABILITY, 3, 1),
          std::vector<uint>{two}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::DIRICHLET,
          ValueType(
              VariableType::COL_SIMPLEX_MATRIX, AtomicType::PROBABILITY, 3, 1),
          std::vector<uint>{cm1, two}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::DIRICHLET,
          ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 1),
          std::vector<uint>{cm1}),
      std::invalid_argument);

  Eigen::MatrixXd m2(2, 2);
  m2 << 1.5, 1.0, 2.0, 1.5;
  uint cm2 = g.add_constant_pos_matrix(m2);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::DIRICHLET,
          ValueType(
              VariableType::COL_SIMPLEX_MATRIX, AtomicType::PROBABILITY, 3, 1),
          std::vector<uint>{cm2}),
      std::invalid_argument);
}

TEST(testdistrib, dirichlet) {
  Graph g;

  Eigen::MatrixXd m1(3, 1);
  m1 << 1.5, 1.0, 2.0;

  uint flat_dist = g.add_distribution(
      DistributionType::FLAT,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::POS_REAL, 3, 1),
      std::vector<uint>{});
  uint flat_sample = g.add_operator(OperatorType::SAMPLE, {flat_dist});
  g.observe(flat_sample, m1);

  uint diri_dist = g.add_distribution(
      DistributionType::DIRICHLET,
      ValueType(
          VariableType::COL_SIMPLEX_MATRIX, AtomicType::PROBABILITY, 3, 1),
      std::vector<uint>{flat_sample});

  // test log prob
  uint diri_sample = g.add_operator(OperatorType::SAMPLE, {diri_dist});
  Eigen::MatrixXd obs(3, 1);
  obs << 0.6, 0.06, 0.34;
  g.observe(diri_sample, obs);
  EXPECT_NEAR(g.full_log_prob(), 1.2403, 0.01);

  // test backward_param() and backward_value()
  // verify with PyTorch
  // param = tensor([1.5, 1., 2.], requires_grad=True)
  // diri = dist.Dirichlet(param)
  // value = tensor([0.6, 0.06, 0.34], requires_grad=True)
  // log_prob = diri.log_prob(value)
  // grad(log_prob, param, retain_graph=True)
  // grad(log_prob, value)
  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 2);
  EXPECT_NEAR(grad1[0]->_matrix(0), 0.8416, 1e-3);
  EXPECT_NEAR(grad1[0]->_matrix(1), -0.8473, 1e-3);
  EXPECT_NEAR(grad1[0]->_matrix(2), -0.1127, 1e-3);
  EXPECT_NEAR(grad1[1]->_matrix(0), 0.8333, 1e-3);
  EXPECT_NEAR(grad1[1]->_matrix(1), 0.0, 1e-3);
  EXPECT_NEAR(grad1[1]->_matrix(2), 2.9412, 1e-3);

  // different shape
  Graph g2;
  flat_dist = g2.add_distribution(
      DistributionType::FLAT,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::POS_REAL, 4, 1),
      std::vector<uint>{});
  uint p1 = g2.add_operator(OperatorType::SAMPLE, {flat_dist});

  uint d1 = g2.add_distribution(
      DistributionType::DIRICHLET,
      ValueType(
          VariableType::COL_SIMPLEX_MATRIX, AtomicType::PROBABILITY, 4, 1),
      std::vector<uint>{p1});
  uint x = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{d1});

  Eigen::MatrixXd p1_obs(4, 1);
  p1_obs << 2., 1., 3., 2.;
  g2.observe(p1, p1_obs);
  Eigen::MatrixXd p2_obs(4, 1);
  Eigen::MatrixXd x_obs(4, 1);
  x_obs << 0.05, 0.05, 0.8, 0.12;
  g2.observe(x, x_obs);

  // PyTorch verification
  // p1 = tensor([2.0, 1., 3., 2.0], requires_grad=True)
  // x = tensor([0.05, 0.05, 0.8, 0.12], requires_grad=True)
  // d1 = dist.Dirichlet(p1)
  // log_p = d1.log_prob(x)
  // grad(log_p, p1, retain_graph=True)
  // grad(log_p, x)
  EXPECT_NEAR(g2.full_log_prob(), 2.2697, 1e-3);
  std::vector<DoubleMatrix*> back_grad;
  g2.eval_and_grad(back_grad);
  EXPECT_EQ(back_grad.size(), 2);
  // p1
  EXPECT_NEAR(back_grad[0]->_matrix(0), -1.4029, 1e-3);
  EXPECT_NEAR(back_grad[0]->_matrix(1), -0.4029, 1e-3);
  EXPECT_NEAR(back_grad[0]->_matrix(2), 0.8697, 1e-3);
  EXPECT_NEAR(back_grad[0]->_matrix(3), -0.5274, 1e-3);
  // x
  EXPECT_NEAR(back_grad[1]->_matrix(0), 20, 1e-3);
  EXPECT_NEAR(back_grad[1]->_matrix(1), 0, 1e-3);
  EXPECT_NEAR(back_grad[1]->_matrix(2), 2.5, 1e-3);
  EXPECT_NEAR(back_grad[1]->_matrix(3), 8.3333, 1e-3);
}

TEST(testdistrib, dirichlet_sample) {
  Graph g;
  Eigen::MatrixXd m1(3, 1);
  m1 << 1.5, 1.0, 2.0;
  uint cm1 = g.add_constant_pos_matrix(m1);
  uint diri_dist = g.add_distribution(
      DistributionType::DIRICHLET,
      ValueType(
          VariableType::COL_SIMPLEX_MATRIX, AtomicType::PROBABILITY, 3, 1),
      std::vector<uint>{cm1});

  // test distribution of mean and variance
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{diri_dist});
  g.query(x);
  std::vector<std::vector<NodeValue>> samples =
      g.infer(100000, InferenceType::REJECTION);
  Eigen::MatrixXd mean(3, 1);
  Eigen::ArrayXd std(3, 1);
  for (int i = 0; i < samples.size(); i++) {
    mean += samples[i][0]._matrix;
  }
  mean /= samples.size();
  EXPECT_NEAR(mean(0), 0.33, 0.01);
  EXPECT_NEAR(mean(1), 0.22, 0.01);
  EXPECT_NEAR(mean(2), 0.44, 0.01);
  for (int i = 0; i < samples.size(); i++) {
    std += (samples[i][0]._matrix - mean).array().pow(2);
  }
  std = (std / (100000 - 1)).cwiseSqrt();
  EXPECT_NEAR(std(0), 0.201, 0.01);
  EXPECT_NEAR(std(1), 0.1773, 0.01);
  EXPECT_NEAR(std(2), 0.2119, 0.01);
}
