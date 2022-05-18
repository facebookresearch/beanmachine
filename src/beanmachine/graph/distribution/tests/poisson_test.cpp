/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"

using namespace beanmachine::graph;

TEST(testdistrib, poisson) {
  Graph g;
  const double A = 5.0;
  const double B = 2.0;
  auto pos1 = g.add_constant_pos_real(A);
  auto pos2 = g.add_constant_pos_real(B);
  // negative test: the sample_type must be natural.
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::POISSON, AtomicType::REAL, std::vector<uint>{pos1}),
      std::invalid_argument);
  // negative test: poisson has exactly one parent
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::POISSON, AtomicType::NATURAL, std::vector<uint>{}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::POISSON,
          AtomicType::NATURAL,
          std::vector<uint>{pos1, pos2}),
      std::invalid_argument);
  // test creation of distribution
  auto poisson_dist = g.add_distribution(
      DistributionType::POISSON, AtomicType::NATURAL, std::vector<uint>{pos1});
  // test distribution of mean and variance
  auto x =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{poisson_dist});
  auto x_pos_real =
      g.add_operator(OperatorType::TO_POS_REAL, std::vector<uint>{x});
  auto x_sq = g.add_operator(
      OperatorType::MULTIPLY, std::vector<uint>{x_pos_real, x_pos_real});
  g.query(x);
  g.query(x_sq);
  const std::vector<double>& means =
      g.infer_mean(10000, InferenceType::REJECTION);
  EXPECT_NEAR(means[0], A, 0.1);
  EXPECT_NEAR(means[1] - means[0] * means[0], A, 0.1);
  // test log_prob
  g.observe(x, (natural_t)2);
  EXPECT_NEAR(g.log_prob(x), -2.4743, 0.001); // value computed from pytorch
  // test gradients of the parameter lambda
  // lambda ~ FLAT
  // y ~ poisson(lambda^2)
  double grad1 = 0;
  double grad2 = 0;

  auto lambda = g.add_operator(
      OperatorType::SAMPLE,
      std::vector<uint>{g.add_distribution(
          DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{})});
  auto lambda_sq =
      g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{lambda, lambda});
  g.observe(lambda, 3.0);
  auto y_dist = g.add_distribution(
      DistributionType::POISSON,
      AtomicType::NATURAL,
      std::vector<uint>{lambda_sq});
  auto y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{y_dist});
  g.observe(y, (natural_t)3);
  // Verified in pytorch using the following code:
  // x = torch.tensor([3.0])
  // lmbda = torch.tensor([3.0], requires_grad=True)
  // f_x = torch.distributions.Poisson(lmbda**2).log_prob(x)
  // f_grad = torch.autograd.grad(f_x, lmbda, create_graph=True)
  // f_grad2 = torch.autograd.grad(f_grad, lmbda)
  // f_grad -> -4.000 f_grad2 -> -2.6667
  grad1 = grad2 = 0;
  g.gradient_log_prob(lambda, grad1, grad2);
  EXPECT_NEAR(grad1, -4.000, 0.001);
  EXPECT_NEAR(grad2, -2.6667, 0.001);
  // test backward_param, backward_value
  // test backward_param_iid, backward_param_iid
  auto two = g.add_constant((natural_t)2);
  auto y2 = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{y_dist, two, two});
  Eigen::MatrixXn m_y(2, 2);
  m_y << 2, 3, 4, 5;
  g.observe(y2, m_y);
  std::vector<DoubleMatrix*> grad;
  g.eval_and_grad(grad);

  // To verify the results for y2 with Pytorch
  // lmbda = torch.tensor([3.0], requires_grad=True)
  // y = torch.tensor([3.0], requires_grad=True)
  // y2 = torch.tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
  // dist = torch.distributions.Poisson(lmbda**2)
  // f = dist.log_prob(y2).sum() + dist.log_prob(y)
  // f_grad_y = torch.autograd.grad(f, y, retain_graph=True)
  // f_grad_lmbda = torch.autograd.grad(f, lmbda, retain_graph=True)
  // f_grad_y2 = torch.autograd.grad(f, y2)
  // f_grad_lmbda -> -18.6667 f_grad_y -> 0.9411 and f_grad_y2 ->
  //   [[1.2744, 0.9411],
  //    [0.6911, 0.4911]]
  // Since we don't evaluate the gradient on discrete values in BMG,
  // the grad of x, y, and y2 should be 0.
  EXPECT_EQ(grad.size(), 4);
  EXPECT_NEAR((*grad[0]), 0, 1e-3); // x, disconnected
  EXPECT_NEAR((*grad[1]), -18.6667, 1e-3); // lambda
  EXPECT_NEAR((*grad[2]), 0, 1e-3); // y
  EXPECT_NEAR(grad[3]->coeff(0, 0), 0, 1e-3); // y2
  EXPECT_NEAR(grad[3]->coeff(0, 1), 0, 1e-3);
  EXPECT_NEAR(grad[3]->coeff(1, 0), 0, 1e-3);
  EXPECT_NEAR(grad[3]->coeff(1, 1), 0, 1e-3);
}
