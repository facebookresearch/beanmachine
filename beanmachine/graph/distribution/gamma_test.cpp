// Copyright (c) Facebook, Inc. and its affiliates.
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
      g.add_distribution(DistributionType::GAMMA, AtomicType::REAL, std::vector<uint>{pos1, pos2}),
      std::invalid_argument);
  // negative tests gamma has two parents
  EXPECT_THROW(
      g.add_distribution(DistributionType::GAMMA, AtomicType::POS_REAL, std::vector<uint>{}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(DistributionType::GAMMA, AtomicType::POS_REAL, std::vector<uint>{pos1}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(DistributionType::GAMMA, AtomicType::POS_REAL, std::vector<uint>{pos1, pos1, pos1}),
      std::invalid_argument);
  // negative test the parents must be both positive real.
  EXPECT_THROW(
      g.add_distribution(DistributionType::GAMMA, AtomicType::POS_REAL, std::vector<uint>{neg1, pos1}),
      std::invalid_argument);
  // test creation of a distribution
  auto gamma_dist = g.add_distribution(
      DistributionType::GAMMA, AtomicType::POS_REAL, std::vector<uint>{pos1, pos2});
  // test distribution of mean and variance
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{gamma_dist});
  auto x_sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{x, x});
  g.query(x);
  g.query(x_sq);
  const std::vector<double>& means = g.infer_mean(100000, InferenceType::REJECTION);
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
  auto shape = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{
      g.add_distribution(DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{})});
  auto shape_sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{shape, shape});
  g.observe(shape, 3.0);
  auto rate = x;
  auto rate_sq = x_sq;
  auto y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{
      g.add_distribution(DistributionType::GAMMA, AtomicType::POS_REAL,
      std::vector<uint>{shape_sq, rate_sq})});
  g.observe(y, 3.0);
  // shape = torch.tensor([3.0], requires_grad=True)
  // rate = torch.tensor([1.5], requires_grad=True)
  // f_x = torch.distributions.Gamma(shape**2, rate**2).log_prob(3.0) + torch.distributions.Gamma(5, 2).log_prob(rate)
  // f_grad_shape = torch.autograd.grad(f_x, shape, create_graph=True)
  // f_grad2_shape = torch.autograd.grad(f_grad_shape, shape)
  // f_grad_shape, f_grad2_shape # -> -1.3866, -4.6926
  // f_grad_rate = torch.autograd.grad(f_x, rate, create_graph=True)
  // f_grad2_rate = torch.autograd.grad(f_grad_rate, rate)
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
}
