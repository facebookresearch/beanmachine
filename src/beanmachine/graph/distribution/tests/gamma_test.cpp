/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"

using namespace beanmachine::graph;

TEST(testdistrib, gamma) {
  Graph g;
  const double A = 5.0;
  const double B = 2.0;
  const double C = -1.0;
  auto pos1 = g.add_constant_pos_real(A);
  auto pos2 = g.add_constant_pos_real(B);
  auto neg1 = g.add_constant(C);
  // negative test the sample_type must be positive real.
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::GAMMA,
          AtomicType::REAL,
          std::vector<uint>{pos1, pos2}),
      std::invalid_argument);
  // negative tests gamma has two parents
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::GAMMA, AtomicType::POS_REAL, std::vector<uint>{}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::GAMMA,
          AtomicType::POS_REAL,
          std::vector<uint>{pos1}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::GAMMA,
          AtomicType::POS_REAL,
          std::vector<uint>{pos1, pos1, pos1}),
      std::invalid_argument);
  // negative test the parents must be both positive real.
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::GAMMA,
          AtomicType::POS_REAL,
          std::vector<uint>{neg1, pos1}),
      std::invalid_argument);
  // test creation of a distribution
  auto gamma_dist = g.add_distribution(
      DistributionType::GAMMA,
      AtomicType::POS_REAL,
      std::vector<uint>{pos1, pos2});
  // test distribution of mean and variance
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{gamma_dist});
  auto x_sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{x, x});
  g.query(x);
  g.query(x_sq);
  const std::vector<double>& means =
      g.infer_mean(100000, InferenceType::REJECTION);
  EXPECT_NEAR(means[0], A / B, 0.1);
  EXPECT_NEAR(means[1] - means[0] * means[0], A / (B * B), 0.1);
  // test log_prob
  g.observe(x, 1.5);
  EXPECT_NEAR(g.log_prob(x), -1.0905, 0.001); // value computed from pytorch
  // test gradient of the sampled value
  // Verified in pytorch using the following code:
  // x = torch.tensor([1.5], requires_grad=True)
  // f_x = torch.distributions.Gamma(5.0, 2.0).log_prob(x)
  // f_grad = torch.autograd.grad(f_x, x, create_graph=True)
  // f_grad2 = torch.autograd.grad(f_grad, x)
  // f_grad -> 0.6667 and f_grad2 -> -1.7778
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(x, grad1, grad2);
  EXPECT_NEAR(grad1, 0.6667, 0.001);
  EXPECT_NEAR(grad2, -1.7778, 0.001);
  // test gradients of the parameters
  // shape ~ FLAT
  // rate ~ Gamma(5, 2)
  // y ~ Gamma(shape^2, rate^2)
  auto shape = g.add_operator(
      OperatorType::SAMPLE,
      std::vector<uint>{g.add_distribution(
          DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{})});
  auto shape_sq =
      g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{shape, shape});
  g.observe(shape, 3.0);
  auto rate = x;
  auto rate_sq = x_sq;
  auto y_dist = g.add_distribution(
      DistributionType::GAMMA,
      AtomicType::POS_REAL,
      std::vector<uint>{shape_sq, rate_sq});
  auto y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{y_dist});
  g.observe(y, 3.0);
  // shape = torch.tensor([3.0], requires_grad=True)
  // rate = torch.tensor([1.5], requires_grad=True)
  // f_x = torch.distributions.Gamma(shape**2, rate**2).log_prob(3.0) +
  // torch.distributions.Gamma(5, 2).log_prob(rate) f_grad_shape =
  // torch.autograd.grad(f_x, shape, create_graph=True) f_grad2_shape =
  // torch.autograd.grad(f_grad_shape, shape) f_grad_shape, f_grad2_shape # ->
  // -1.3866, -4.6926 f_grad_rate = torch.autograd.grad(f_x, rate,
  // create_graph=True) f_grad2_rate = torch.autograd.grad(f_grad_rate, rate)
  // f_grad_rate, f_grad2_rate # -> 3.6667, -15.7778
  // we will call gradient_log_prob from all the parameters one time
  // to ensure that their children are evaluated
  g.gradient_log_prob(shape, grad1, grad2);
  g.gradient_log_prob(rate, grad1, grad2);
  // test shape
  grad1 = grad2 = 0;
  g.gradient_log_prob(shape, grad1, grad2);
  EXPECT_NEAR(grad1, -1.3866, 0.001);
  EXPECT_NEAR(grad2, -4.6926, 0.001);
  // test rate
  grad1 = grad2 = 0;
  g.gradient_log_prob(rate, grad1, grad2);
  EXPECT_NEAR(grad1, 3.6667, 0.001);
  EXPECT_NEAR(grad2, -15.7778, 0.001);
  // test test backward_param, backward_value, backward_param_iid,
  // backward_value_iid
  auto two = g.add_constant((natural_t)2);
  auto y2 = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{y_dist, two, two});
  Eigen::MatrixXd m_y(2, 2);
  m_y << 4.1, 3.2, 2.3, 1.4;
  g.observe(y2, m_y);
  std::vector<DoubleMatrix*> grad;
  g.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 4);
  EXPECT_NEAR((*grad[0]), 18.6667, 1e-3); // rate
  EXPECT_NEAR((*grad[1]), -10.8386, 1e-3); // shape
  EXPECT_NEAR((*grad[2]), 0.4167, 1e-3); // y
  EXPECT_NEAR(grad[3]->_matrix.coeff(0), -0.2988, 1e-3); // y2
  EXPECT_NEAR(grad[3]->_matrix.coeff(1), 1.2283, 1e-3);
  EXPECT_NEAR(grad[3]->_matrix.coeff(2), 0.2500, 1e-3);
  EXPECT_NEAR(grad[3]->_matrix.coeff(3), 3.4643, 1e-3);

  // test sample/iid_sample from a mixture of gammas
  Graph g2;
  auto size = g2.add_constant((natural_t)2);
  auto flat_pos = g2.add_distribution(
      DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{});
  auto flat_prob = g2.add_distribution(
      DistributionType::FLAT, AtomicType::PROBABILITY, std::vector<uint>{});
  auto rate1 =
      g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_pos});
  auto shape1 =
      g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_pos});
  auto rate2 =
      g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_pos});
  auto shape2 =
      g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_pos});
  auto d1 = g2.add_distribution(
      DistributionType::GAMMA,
      AtomicType::POS_REAL,
      std::vector<uint>{rate1, shape1});
  auto d2 = g2.add_distribution(
      DistributionType::GAMMA,
      AtomicType::POS_REAL,
      std::vector<uint>{rate2, shape2});
  auto p = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_prob});
  auto dist = g2.add_distribution(
      DistributionType::BIMIXTURE,
      AtomicType::POS_REAL,
      std::vector<uint>{p, d1, d2});
  auto x1 = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist});
  auto x2 =
      g2.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{dist, size});
  g2.observe(rate1, 10.0);
  g2.observe(shape1, 1.2);
  g2.observe(rate2, 0.4);
  g2.observe(shape2, 1.8);
  g2.observe(p, 0.37);
  g2.observe(x1, 2.5);
  Eigen::MatrixXd xobs(2, 1);
  xobs << 0.5, 1.5;
  g2.observe(x2, xobs);
  // To verify the results with pyTorch:
  // a1 = torch.tensor(10.0, requires_grad=True)
  // b1 = torch.tensor(1.2, requires_grad=True)
  // a2 = torch.tensor(0.4, requires_grad=True)
  // b2 = torch.tensor(1.8, requires_grad=True)
  // p = torch.tensor(0.37, requires_grad=True)
  // x = torch.tensor([2.5, 0.5, 1.5], requires_grad=True)
  // d1 = torch.distributions.Gamma(a1, b1)
  // d2 = torch.distributions.Gamma(a2, b2)
  // f1 = d1.log_prob(x).exp()
  // f2 = d2.log_prob(x).exp()
  // log_p = (p * f1 + (tensor(1.0) - p) * f2).log().sum()
  // torch.autograd.grad(log_p, x)[0]
  EXPECT_NEAR(g2.full_log_prob(), -11.1268, 1e-3);
  std::vector<DoubleMatrix*> back_grad;
  g2.eval_and_grad(back_grad);
  EXPECT_EQ(back_grad.size(), 7);
  EXPECT_NEAR((*back_grad[0]), -0.3983, 1e-3); // rate1
  EXPECT_NEAR((*back_grad[1]), 2.0114, 1e-3); // shape1
  EXPECT_NEAR((*back_grad[2]), 8.6768, 1e-3); // rate2
  EXPECT_NEAR((*back_grad[3]), -3.0509, 1e-3); // shape2
  EXPECT_NEAR((*back_grad[4]), -3.2842, 1e-3); // p
  EXPECT_NEAR((*back_grad[5]), -0.5200, 1e-3); // x1
  EXPECT_NEAR(back_grad[6]->_matrix.coeff(0), -3.0000, 1e-3); // x2
  EXPECT_NEAR(back_grad[6]->_matrix.coeff(1), -2.1852, 1e-3);
}
