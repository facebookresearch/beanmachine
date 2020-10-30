// Copyright (c) Facebook, Inc. and its affiliates.
#include <gtest/gtest.h>

#include "beanmachine/graph/distribution/beta.h"
#include "beanmachine/graph/graph.h"

using namespace beanmachine::graph;

TEST(testdistrib, beta1) {
  Graph g1;
  const double A = 1.1;
  const double B = 5.0;
  const double C = -1.5;
  auto pos1 = g1.add_constant_pos_real(A);
  auto pos2 = g1.add_constant_pos_real(B);
  auto neg1 = g1.add_constant(C);
  // negative test the sample_type must be probability.
  EXPECT_THROW(
      g1.add_distribution(
          DistributionType::BETA,
          AtomicType::REAL,
          std::vector<uint>{pos1, pos2}),
      std::invalid_argument);
  // negative tests gamma has two parents
  EXPECT_THROW(
      g1.add_distribution(
          DistributionType::BETA, AtomicType::PROBABILITY, std::vector<uint>{}),
      std::invalid_argument);
  EXPECT_THROW(
      g1.add_distribution(
          DistributionType::BETA,
          AtomicType::PROBABILITY,
          std::vector<uint>{pos1}),
      std::invalid_argument);
  // negative test the parents must be both positive real.
  EXPECT_THROW(
      g1.add_distribution(
          DistributionType::BETA,
          AtomicType::PROBABILITY,
          std::vector<uint>{neg1, pos1}),
      std::invalid_argument);
  // test creation of a distribution
  auto beta_dist = g1.add_distribution(
      DistributionType::BETA,
      AtomicType::PROBABILITY,
      std::vector<uint>{pos1, pos2});
  // test distribution of mean and variance
  auto x = g1.add_operator(OperatorType::SAMPLE, std::vector<uint>{beta_dist});
  auto x_sq = g1.add_operator(OperatorType::MULTIPLY, std::vector<uint>{x, x});
  g1.query(x);
  g1.query(x_sq);
  const std::vector<double>& means =
      g1.infer_mean(100000, InferenceType::REJECTION);
  EXPECT_NEAR(means[0], A / (A + B), 0.01);
  EXPECT_NEAR(
      means[1] - means[0] * means[0],
      A * B / ((A + B) * (A + B) * (A + B + 1)),
      0.01);
  // test log_prob
  g1.observe(x, 0.2);
  EXPECT_NEAR(g1.log_prob(x), 0.7773, 0.001); // value computed from pytorch
  // test gradient of the sampled value
  // Verified in pytorch using the following code:
  // x = torch.tensor([0.2], requires_grad=True)
  // f_x = torch.distributions.Beta(1.1, 5.0).log_prob(x)
  // f_grad = torch.autograd.grad(f_x, x, create_graph=True)
  // f_grad2 = torch.autograd.grad(f_grad, x)
  // f_grad -> -4.5 and f_grad2 -> -8.75
  double grad1 = 0;
  double grad2 = 0;
  g1.gradient_log_prob(x, grad1, grad2);
  EXPECT_NEAR(grad1, -4.5, 0.001);
  EXPECT_NEAR(grad2, -8.75, 0.001);

  // test gradients of the parameters
  // a ~ FLAT
  // b ~ FLAT
  // y ~ Beta(a^2, b^2)
  Graph g2;
  auto a = g2.add_operator(
      OperatorType::SAMPLE,
      std::vector<uint>{g2.add_distribution(
          DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{})});
  auto a_sq = g2.add_operator(OperatorType::MULTIPLY, std::vector<uint>{a, a});
  g2.observe(a, 1.5);
  auto b = g2.add_operator(
      OperatorType::SAMPLE,
      std::vector<uint>{g2.add_distribution(
          DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{})});
  auto b_sq = g2.add_operator(OperatorType::MULTIPLY, std::vector<uint>{b, b});
  g2.observe(b, 2.0);
  auto y = g2.add_operator(
      OperatorType::SAMPLE,
      std::vector<uint>{g2.add_distribution(
          DistributionType::BETA,
          AtomicType::PROBABILITY,
          std::vector<uint>{a_sq, b_sq})});
  g2.query(y);
  // eval the graph from two roots
  g2.log_prob(a);
  g2.log_prob(b);
  g2.observe(y, 0.3);
  // a = torch.tensor([1.5], requires_grad=True)
  // b = torch.tensor([2.0], requires_grad=True)
  // f_x = torch.distributions.Beta(a**2, b**2).log_prob(tensor(0.3))
  // f_grad_a = torch.autograd.grad(f_x, a, create_graph=True)
  // f_grad2_a = torch.autograd.grad(f_grad_a, a)
  // f_grad_a[0].item(), f_grad2_a[0].item() # -> -0.07819676399230957,
  // -3.506779909133911 f_grad_b = torch.autograd.grad(f_x, b,
  // create_graph=True) f_grad2_b = torch.autograd.grad(f_grad_b, b)
  // f_grad_b[0].item(), f_grad2_b[0].item() # -> 0.5506436824798584,
  // -1.4901777505874634

  // grad w.r.t a
  grad1 = grad2 = 0;
  g2.gradient_log_prob(a, grad1, grad2);
  EXPECT_NEAR(grad1, -0.07819, 0.001);
  EXPECT_NEAR(grad2, -3.50677, 0.001);
  // grad w.r.t. b
  grad1 = grad2 = 0;
  g2.gradient_log_prob(b, grad1, grad2);
  EXPECT_NEAR(grad1, 0.55064, 0.001);
  EXPECT_NEAR(grad2, -1.49017, 0.001);
}

TEST(testdistrib, beta2) {
  Graph g1;
  const double A = 1.1;
  const double B = 5.0;
  auto pos1 = g1.add_constant_pos_real(A);
  auto pos2 = g1.add_constant_pos_real(B);
  auto two = g1.add_constant((natural_t)2);

  auto beta_dist = g1.add_distribution(
      DistributionType::BETA,
      AtomicType::PROBABILITY,
      std::vector<uint>{pos1, pos2});
  // test iid_sample related operations
  auto x = g1.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{beta_dist, two});
  g1.query(x);
  Eigen::MatrixXd matrix1(2, 1);
  matrix1 << 0.6, 0.5;
  g1.observe(x, matrix1);
  EXPECT_NEAR(g1.log_prob(x), -2.8965, 0.001); // value computed from pytorch

  // TODO: update the tests below with backward gradient computation.
  // test gradient of the sampled value
  // Verified in pytorch using the following code:
  // x = torch.tensor([0.6], requires_grad=True)
  // # x = torch.tensor([0.5], requires_grad=True)
  // f_x = torch.distributions.Beta(1.1, 5.0).log_prob(x)
  // f_grad = torch.autograd.grad(f_x, x, create_graph=True)
  // f_grad2 = torch.autograd.grad(f_grad, x)

  // test gradients of the parameters
  // a ~ FLAT
  // b ~ FLAT
  // y = (y1, y2) ~ Beta(a^2, b^2)
  Graph g2;
  auto nat_node = g2.add_constant((natural_t)2);
  auto a = g2.add_operator(
      OperatorType::SAMPLE,
      std::vector<uint>{g2.add_distribution(
          DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{})});
  auto a_sq = g2.add_operator(OperatorType::MULTIPLY, std::vector<uint>{a, a});
  g2.observe(a, 1.5);
  auto b = g2.add_operator(
      OperatorType::SAMPLE,
      std::vector<uint>{g2.add_distribution(
          DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{})});
  auto b_sq = g2.add_operator(OperatorType::MULTIPLY, std::vector<uint>{b, b});
  g2.observe(b, 2.0);
  auto y = g2.add_operator(
      OperatorType::IID_SAMPLE,
      std::vector<uint>{g2.add_distribution(
                            DistributionType::BETA,
                            AtomicType::PROBABILITY,
                            std::vector<uint>{a_sq, b_sq}),
                        nat_node});
  g2.query(y);
  // eval the graph from two roots
  g2.log_prob(a);
  g2.log_prob(b);
  Eigen::MatrixXd matrix2(2, 1);
  matrix2 << 0.3, 0.4;
  g2.observe(y, matrix2);
  // a = torch.tensor([1.5], requires_grad=True)
  // b = torch.tensor([2.0], requires_grad=True)
  // f_x = torch.distributions.Beta(a**2, b**2).log_prob(tensor([0.3,
  // 0.4])).sum() f_grad_a = torch.autograd.grad(f_x, a, create_graph=True)
  // f_grad2_a = torch.autograd.grad(f_grad_a, a)
  // f_grad_a[0].item(), f_grad2_a[0].item() # -> 0.7066526412963867,
  // -6.438195705413818 f_grad_b = torch.autograd.grad(f_x, b,
  // create_graph=True) f_grad2_b = torch.autograd.grad(f_grad_b, b)
  // f_grad_b[0].item(), f_grad2_b[0].item() # -> 0.48468494415283203,
  // -3.288656711578369

  // grad w.r.t a
  // grad w.r.t. b
}
