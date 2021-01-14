// Copyright (c) Facebook, Inc. and its affiliates.
#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"

using namespace beanmachine::graph;

TEST(testdistrib, bimixture) {
  Graph g1;
  auto real1 = g1.add_constant(-0.5);
  auto real2 = g1.add_constant(0.5);
  auto pos1 = g1.add_constant_pos_real(1.5);
  auto pos2 = g1.add_constant_pos_real(2.5);
  auto prob1 = g1.add_constant_probability(0.3);
  auto gamma_dist = g1.add_distribution(
      DistributionType::GAMMA,
      AtomicType::POS_REAL,
      std::vector<uint>{pos1, pos2});
  auto normal_dist1 = g1.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real1, pos1});
  auto normal_dist2 = g1.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real2, pos2});
  // negative test: must have three parents.
  EXPECT_THROW(
      g1.add_distribution(
          DistributionType::BIMIXTURE, AtomicType::REAL, std::vector<uint>{}),
      std::invalid_argument);
  // negative tests: first parent must be probability
  EXPECT_THROW(
      g1.add_distribution(
          DistributionType::BIMIXTURE,
          AtomicType::REAL,
          std::vector<uint>{real1, normal_dist1, normal_dist2}),
      std::invalid_argument);
  // negative test: 2nd, 3rd parent must be distributions.
  EXPECT_THROW(
      g1.add_distribution(
          DistributionType::BIMIXTURE,
          AtomicType::REAL,
          std::vector<uint>{prob1, real1, normal_dist2}),
      std::invalid_argument);
  EXPECT_THROW(
      g1.add_distribution(
          DistributionType::BIMIXTURE,
          AtomicType::REAL,
          std::vector<uint>{prob1, normal_dist1, real1}),
      std::invalid_argument);
  // negative test: sample type must be consistent
  EXPECT_THROW(
      g1.add_distribution(
          DistributionType::BIMIXTURE,
          AtomicType::REAL,
          std::vector<uint>{prob1, gamma_dist, normal_dist1}),
      std::invalid_argument);
  EXPECT_THROW(
      g1.add_distribution(
          DistributionType::BIMIXTURE,
          AtomicType::POS_REAL,
          std::vector<uint>{prob1, normal_dist1, normal_dist2}),
      std::invalid_argument);
  // test creation of a distribution
  auto bimix_dist = g1.add_distribution(
      DistributionType::BIMIXTURE,
      AtomicType::REAL,
      std::vector<uint>{prob1, normal_dist1, normal_dist2});
  // test distribution of mean and variance
  auto x = g1.add_operator(OperatorType::SAMPLE, std::vector<uint>{bimix_dist});
  auto x_sq = g1.add_operator(OperatorType::MULTIPLY, std::vector<uint>{x, x});
  g1.query(x);
  g1.query(x_sq);
  const std::vector<double>& means =
      g1.infer_mean(100000, InferenceType::REJECTION);
  EXPECT_NEAR(means[0], 0.2, 0.01);
  EXPECT_NEAR(means[1] - means[0] * means[0], 5.26, 0.1);
  // test log_prob
  g1.observe(x, 1.5);
  EXPECT_NEAR(g1.log_prob(x), -1.9957, 0.001); // value computed from pytorch
  // test gradient of the sampled value
  // Verified in pytorch using the following code:
  // d1 = torch.distributions.Normal(tensor(-0.5), tensor(1.5))
  // d2 = torch.distributions.Normal(tensor(0.5), tensor(2.5))
  // p = tensor(0.3)
  // x = tensor(1.5, requires_grad=True)
  // f1 = d1.log_prob(x).exp()
  // f2 = d2.log_prob(x).exp()
  // f_x = (p * f1 + (tensor(1) - p) * f2).log()
  // f_grad = torch.autograd.grad(f_x, x, create_graph=True)
  // f_grad2 = torch.autograd.grad(f_grad, x)
  double grad1 = 0;
  double grad2 = 0;
  g1.gradient_log_prob(x, grad1, grad2);
  EXPECT_NEAR(grad1, -0.3359, 0.001);
  EXPECT_NEAR(grad2, -0.1314, 0.001);
  // test backward_value
  std::vector<DoubleMatrix*> grad;
  g1.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 1);
  EXPECT_NEAR(grad[0]->_double, -0.3359, 1e-3);

  // test gradients of the parameters
  // a ~ FLAT(REAL), b ~ FLAT(REAL), c ~ FLAT(POS_REAL)
  // p = LOGISTIC(a)
  // f1 = NORMAL(0, c^2), f2 = NORMAL(b^2, c^2)
  // x ~ BIMIXTURE(p, f1, f2)
  Graph g2;
  auto flat_real = g2.add_distribution(
      DistributionType::FLAT, AtomicType::REAL, std::vector<uint>{});
  auto flat_pos_real = g2.add_distribution(
      DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{});
  auto a = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_real});
  auto b = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_real});
  auto c =
      g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_pos_real});
  auto p = g2.add_operator(OperatorType::LOGISTIC, std::vector<uint>{a});
  g2.observe(a, -1.5);
  auto zero = g2.add_constant(0.0);
  auto b_sq = g2.add_operator(OperatorType::MULTIPLY, std::vector<uint>{b, b});
  g2.observe(b, 1.8);
  auto c_sq = g2.add_operator(OperatorType::MULTIPLY, std::vector<uint>{c, c});
  g2.observe(c, 1.5);
  auto dist1 = g2.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{zero, c_sq});
  auto dist2 = g2.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{b_sq, c_sq});
  auto bimix = g2.add_distribution(
      DistributionType::BIMIXTURE,
      AtomicType::REAL,
      std::vector<uint>{p, dist1, dist2});
  auto y = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{bimix});
  g2.query(y);
  const std::vector<double>& ymean = g2.infer_mean(10, InferenceType::NMC);
  EXPECT_NEAR(ymean[0], 2.0, 2.0);
  g2.observe(y, 4.0);
  EXPECT_NEAR(g2.full_log_prob(), -1.9408, 1e-3);
  // a = torch.tensor(-1.5, requires_grad=True)
  // b = torch.tensor(1.8, requires_grad=True)
  // c = torch.tensor(1.5, requires_grad=True)
  // x = torch.tensor(4.0, requires_grad=True)
  // d1 = torch.distributions.Normal(tensor(0.0), c**2)
  // d2 = torch.distributions.Normal(b**2, c**2)
  // f1 = d1.log_prob(x).exp()
  // f2 = d2.log_prob(x).exp()
  // p = torch.sigmoid(a)
  // f_x = (p * f1 + (tensor(1) - p) * f2).log()
  // f_grad_a = torch.autograd.grad(f_x, a, create_graph=True)
  // f_grad2_a = torch.autograd.grad(f_grad_a, a)
  //   -> -0.13603681325912476 -0.10490967333316803
  // f_grad_b = torch.autograd.grad(f_x, b, create_graph=True)
  // f_grad2_b = torch.autograd.grad(f_grad_b, b)
  //   -> 0.5153740644454956 -2.142005205154419
  // f_grad_c = torch.autograd.grad(f_x, c, create_graph=True)
  // f_grad2_c = torch.autograd.grad(f_grad_c, c)
  //   -> -0.9927835464477539 0.48357468843460083

  // we will call gradient_log_prob from all the parameters one time
  // to ensure that their children are evaluated
  g2.gradient_log_prob(a, grad1, grad2);
  g2.gradient_log_prob(b, grad1, grad2);
  g2.gradient_log_prob(c, grad1, grad2);
  // test a
  grad1 = grad2 = 0;
  g2.gradient_log_prob(a, grad1, grad2);
  EXPECT_NEAR(grad1, -0.1360, 0.001);
  EXPECT_NEAR(grad2, -0.1049, 0.001);
  // test b
  grad1 = grad2 = 0;
  g2.gradient_log_prob(b, grad1, grad2);
  EXPECT_NEAR(grad1, 0.5153, 0.001);
  EXPECT_NEAR(grad2, -2.1420, 0.001);
  // test c
  grad1 = grad2 = 0;
  g2.gradient_log_prob(c, grad1, grad2);
  EXPECT_NEAR(grad1, -0.9928, 0.001);
  EXPECT_NEAR(grad2, 0.4835, 0.001);

  // tests for backward methods shall be added to each distribution type
}
