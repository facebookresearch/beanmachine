// Copyright (c) Facebook, Inc. and its affiliates.
#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"

using namespace beanmachine::graph;

TEST(testdistrib, half_normal) {
  Graph g;
  const double MEAN = -11.0;
  const double STD = 3.0;
  auto real1 = g.add_constant(MEAN);
  auto pos1 = g.add_constant_pos_real(STD);
  // negative tests half_normal has two parents
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::HALF_NORMAL, AtomicType::REAL, std::vector<uint>{}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::HALF_NORMAL,
          AtomicType::REAL,
          std::vector<uint>{real1}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::HALF_NORMAL,
          AtomicType::REAL,
          std::vector<uint>{real1, pos1, real1}),
      std::invalid_argument);
  // negative test the parents must be a real and a positive
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::HALF_NORMAL,
          AtomicType::REAL,
          std::vector<uint>{real1, real1}),
      std::invalid_argument);
  // test creation of a distribution
  auto half_normal_dist = g.add_distribution(
      DistributionType::HALF_NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real1, pos1});
  // test distribution of mean and variance
  auto real_val =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{half_normal_dist});
  auto real_sq_val = g.add_operator(
      OperatorType::MULTIPLY, std::vector<uint>{real_val, real_val});
  g.query(real_val);
  g.query(real_sq_val);
  const std::vector<double>& means =
      g.infer_mean(100000, InferenceType::REJECTION);
  EXPECT_NEAR(means[0], MEAN, 0.1);
  EXPECT_NEAR(means[1] - means[0] * means[0], STD * STD, 0.1);
  // test log_prob
  g.observe(real_val, 1.0);
  EXPECT_NEAR(
      g.log_prob(real_val), -10.0176, 0.001); // value computed from pytorch
  // test gradient of log_prob w.r.t. value and the mean
  // f(x) = -.5 (x + 11)^2 / 9  -.5 (x^2 - 3)^2/9
  // f'(x) = -(x + 11) / 9 - (x^2 - 3) (2x) / 9
  // f''(x) = -1/9 - 4x^2/9 - 2 (x^2 - 3) /9
  auto half_normal_dist2 = g.add_distribution(
      DistributionType::HALF_NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real_sq_val, pos1});
  auto real_val2 = g.add_operator(
      OperatorType::SAMPLE, std::vector<uint>{half_normal_dist2});
  g.observe(real_val2, 3.0);
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(real_val, grad1, grad2);
  EXPECT_NEAR(grad1, -0.88888, 0.01);
  EXPECT_NEAR(grad2, -0.11111, 0.01);
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
  auto half_normal_dist3 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real1, pos_sq_val});
  auto real_val3 = g.add_operator(
      OperatorType::SAMPLE, std::vector<uint>{half_normal_dist3});
  g.observe(real_val3, 5.0);
  grad1 = grad2 = 0;
  g.gradient_log_prob(pos_val, grad1, grad2);
  // f(x) = -log(pi/2) + log(3.5) - log(3.5^2 + x^2) -0.5 log(2pi) -log(x^2)
  // -0.5*(5 + 11)^2/x^4 f'(x) = -2x/(3.5^2 + x^2) -2/x +0.5*16^2*4/x^5 f'(7) =
  // -0.483822 f''(x) = -2/(3.5^2 + x^2) + 4x^2/(3.5^2 + x^2)^2   +2/x^2
  // -0.5*16^2*4*5/x^6 f''(7) = 0.038648
  EXPECT_NEAR(grad1, -0.483822, 1e-6);
  EXPECT_NEAR(grad2, 0.038648, 1e-6);
}

TEST(testdistrib, backward_half_normal_half_ormal) {
  Graph g;
  uint zero = g.add_constant(0.0);
  uint pos_one = g.add_constant_pos_real(1.0);
  uint two = g.add_constant((natural_t)2);

  uint half_normal_dist = g.add_distribution(
      DistributionType::HALF_NORMAL,
      AtomicType::REAL,
      std::vector<uint>{zero, pos_one});
  uint mu =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{half_normal_dist});

  uint dist_y = g.add_distribution(
      DistributionType::HALF_NORMAL,
      AtomicType::REAL,
      std::vector<uint>{mu, pos_one});
  uint y =
      g.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{dist_y, two});
  g.observe(mu, 0.1);
  Eigen::MatrixXd yobs(2, 1);
  yobs << 0.5, -0.5;
  g.observe(y, yobs);

  // test log_prob() on vector value:
  double log_prob_y = g.log_prob(y);
  EXPECT_NEAR(log_prob_y, -2.0979, 0.001);
  // test backward_param(), backward_value() and
  // backward_param_iid(), backward_value_iid():
  // To verify the grad1 results with pyTorch:
  // mu = tensor([0.1], requires_grad=True)
  // y = tensor([0.5, -0.5], requires_grad=True)
  // log_p = (
  //     dist.Half_Normal(mu, tensor(1.0)).log_prob(y).sum() +
  //     dist.Half_Normal(tensor(0.0), tensor(1.0)).log_prob(mu)
  // )
  // torch.autograd.grad(log_p, mu) -> -0.3
  // torch.autograd.grad(log_p, y) -> [-0.4, 0.6]
  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 2);
  EXPECT_NEAR(grad1[0]->_double, -0.3, 1e-3);
  EXPECT_NEAR(grad1[1]->_matrix.coeff(0), -0.4, 1e-3);
  EXPECT_NEAR(grad1[1]->_matrix.coeff(1), 0.6, 1e-3);

  // mixture of half_normals
  Graph g2;
  auto size = g2.add_constant((natural_t)2);
  auto flat_real = g2.add_distribution(
      DistributionType::FLAT, AtomicType::REAL, std::vector<uint>{});
  auto flat_pos = g2.add_distribution(
      DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{});
  auto flat_prob = g2.add_distribution(
      DistributionType::FLAT, AtomicType::PROBABILITY, std::vector<uint>{});
  auto m1 = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_real});
  auto m2 = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_real});
  auto s = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_pos});
  auto d1 = g2.add_distribution(
      DistributionType::HALF_NORMAL,
      AtomicType::REAL,
      std::vector<uint>{m1, s});
  auto d2 = g2.add_distribution(
      DistributionType::HALF_NORMAL,
      AtomicType::REAL,
      std::vector<uint>{m2, s});
  auto p = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_prob});
  auto dist = g2.add_distribution(
      DistributionType::BIMIXTURE,
      AtomicType::REAL,
      std::vector<uint>{p, d1, d2});
  auto x = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist});
  auto xiid =
      g2.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{dist, size});
  g2.observe(m1, -1.2);
  g2.observe(m2, 0.4);
  g2.observe(s, 1.8);
  g2.observe(p, 0.37);
  g2.observe(x, -0.5);
  Eigen::MatrixXd xobs(2, 1);
  xobs << 0.5, -1.5;
  g2.observe(xiid, xobs);
  // To verify the results with pyTorch:
  // m1 = torch.tensor(-1.2, requires_grad=True)
  // m2 = torch.tensor(0.4, requires_grad=True)
  // s = torch.tensor(1.8, requires_grad=True)
  // p = torch.tensor(0.37, requires_grad=True)
  // x = torch.tensor([-0.5, 0.5, -1.5], requires_grad=True)
  // d1 = torch.distributions.Half_Normal(m1, s)
  // d2 = torch.distributions.Half_Normal(m2, s)
  // f1 = d1.log_prob(x).exp()
  // f2 = d2.log_prob(x).exp()
  // log_p = (p * f1 + (tensor(1.0) - p) * f2).log().sum()
  // torch.autograd.grad(log_p, x)[0]
  EXPECT_NEAR(g2.full_log_prob(), -5.0911, 1e-3);
  std::vector<DoubleMatrix*> back_grad;
  g2.eval_and_grad(back_grad);
  EXPECT_EQ(back_grad.size(), 6);
  EXPECT_NEAR(back_grad[0]->_double, 0.1794, 1e-3); // m1
  EXPECT_NEAR(back_grad[1]->_double, -0.4410, 1e-3); // m2
  EXPECT_NEAR(back_grad[2]->_double, -1.0964, 1e-3); // s
  EXPECT_NEAR(back_grad[3]->_double, 0.2054, 1e-3); // p
  EXPECT_NEAR(back_grad[4]->_double, 0.0893, 1e-3); // x
  EXPECT_NEAR(back_grad[5]->_matrix.coeff(0), -0.1660, 1e-3); // xiid
  EXPECT_NEAR(back_grad[5]->_matrix.coeff(1), 0.3381, 1e-3);
}
