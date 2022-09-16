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

#include "beanmachine/graph/graph.h"

using namespace beanmachine::graph;

TEST(testdistrib, lognormal) {
  Graph g;
  // LOG_MEAN and LOG_STD are the mean and standard deviation of the log of
  // lognormal distribution
  const double LOG_MEAN = -11.0;
  const double LOG_STD = 3.0;
  const double LOG_STD_SQ = 9.0;
  const double MEAN = std::exp(LOG_MEAN + LOG_STD * LOG_STD / 2);
  auto real1 = g.add_constant_real(LOG_MEAN);
  auto pos1 = g.add_constant_pos_real(LOG_STD);

  // negative tests that log normal has two parents
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::LOG_NORMAL,
          AtomicType::POS_REAL,
          std::vector<uint>{}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::LOG_NORMAL,
          AtomicType::POS_REAL,
          std::vector<uint>{real1}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::LOG_NORMAL,
          AtomicType::POS_REAL,
          std::vector<uint>{real1, pos1, real1}),
      std::invalid_argument);

  // negative test the parents must be a real and a positive
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::LOG_NORMAL,
          AtomicType::POS_REAL,
          std::vector<uint>{real1, real1}),
      std::invalid_argument);

  // test creation of a distribution
  auto log_normal_dist = g.add_distribution(
      DistributionType::LOG_NORMAL,
      AtomicType::POS_REAL,
      std::vector<uint>{real1, pos1});

  // test distribution of mean and variance
  auto real_val =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{log_normal_dist});
  auto real_sq_val = g.add_operator(
      OperatorType::MULTIPLY, std::vector<uint>{real_val, real_val});
  g.query(real_val);
  g.query(real_sq_val);
  const std::vector<double>& means =
      g.infer_mean(100000, InferenceType::REJECTION);
  double variance = (std::exp(LOG_STD_SQ) - 1) * exp(2 * LOG_MEAN + LOG_STD_SQ);
  EXPECT_NEAR(means[0], MEAN, 0.1);
  EXPECT_NEAR(means[1] - means[0] * means[0], variance, 0.1);

  // test log_prob and gradients
  g.observe(real_val, M_E);

  // f(x) = - log(3) - 0.5 log(2*pi) - 0.5 (log(x) + 11)^2 / 3^2 - log(x)
  // f'(x) = (-11 - log(x) - 3^2) / (x * 3^2)
  // f''(x) = (3^2 + log(x) + 11 - 1) / (3^2 * x^2)
  // f(e) = approx. -11.0176;
  // f'(e) = -7/3e (approx. -0.85838 )
  // f''(e) = 20/9e^2 (approx. 0.30074)

  EXPECT_NEAR(g.log_prob(real_val), -11.0176, 0.001);
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(real_val, grad1, grad2);
  EXPECT_NEAR(grad1, -0.85838, 0.01);
  EXPECT_NEAR(grad2, 0.30074, 0.01);

  // test gradient of log_prob w.r.t. sigma
  const double SCALE = 3.5;
  auto pos_scale = g.add_constant_pos_real(SCALE);
  auto half_cauchy_dist = g.add_distribution(
      DistributionType::HALF_CAUCHY,
      AtomicType::POS_REAL,
      std::vector<uint>{pos_scale});
  auto pos_val =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{half_cauchy_dist});
  g.observe(pos_val, 7.0);
  auto pos_sq_val = g.add_operator(
      OperatorType::MULTIPLY, std::vector<uint>{pos_val, pos_val});
  auto normal_dist3 = g.add_distribution(
      DistributionType::LOG_NORMAL,
      AtomicType::POS_REAL,
      std::vector<uint>{real1, pos_sq_val});
  auto real_val3 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{normal_dist3});
  g.observe(real_val3, std::exp(5.0));
  grad1 = grad2 = 0;
  g.gradient_log_prob(pos_val, grad1, grad2);
  EXPECT_NEAR(grad1, -0.483822, 1e-6);
  EXPECT_NEAR(grad2, 0.038648, 1e-6);
}

TEST(testdistrib, backward_lognormal_lognormal) {
  Graph g;
  uint zero = g.add_constant_real(0.0);
  uint pos_one = g.add_constant_pos_real(1.0);
  uint two = g.add_constant_natural(2);

  uint lognormal_dist = g.add_distribution(
      DistributionType::LOG_NORMAL,
      AtomicType::POS_REAL,
      std::vector<uint>{zero, pos_one});
  uint pos_mu =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{lognormal_dist});
  uint mu = g.add_operator(OperatorType::TO_REAL, std::vector<uint>{pos_mu});

  uint dist_y = g.add_distribution(
      DistributionType::LOG_NORMAL,
      AtomicType::POS_REAL,
      std::vector<uint>{mu, pos_one});
  uint y =
      g.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{dist_y, two});
  g.observe(pos_mu, 0.1);
  Eigen::MatrixXd yobs(2, 1);
  yobs << std::exp(0.5), std::exp(-0.5);
  g.observe(y, yobs);

  // test backward_param(), backward_value() and
  // backward_param_iid(), backward_value_iid():
  // To verify the grad1 results with pyTorch:
  // mu = torch.tensor([0.1], requires_grad=True)
  // y = torch.tensor([math.e**0.5, math.e**-0.5], requires_grad=True)
  // log_p = (
  //   torch.distributions.LogNormal(mu, torch.tensor(1.0)).log_prob(y).sum() +
  //   torch.distributions.LogNormal(torch.tensor(0.0),
  //   torch.tensor(1.0)).log_prob(mu)
  // )
  // torch.autograd.grad(log_p, mu) -> 12.8259
  // torch.autograd.grad(log_p, y) -> [-0.8491, -0.6595]
  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 2);
  EXPECT_NEAR((*grad1[0]), 12.8259, 1e-3);
  EXPECT_NEAR(grad1[1]->coeff(0), -0.8491, 1e-3);
  EXPECT_NEAR(grad1[1]->coeff(1), -0.6595, 1e-3);

  // test log_prob() on vector value:
  double log_prob_y = g.log_prob(y);
  EXPECT_NEAR(log_prob_y, -2.0979, 0.001);
}
