// Copyright (c) Facebook, Inc. and its affiliates.
#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"

using namespace beanmachine;
using namespace beanmachine::graph;

TEST(testoperator, complement) {
  // negative test num args can't be zero
  EXPECT_THROW(
      oper::Operator onode1(OperatorType::COMPLEMENT, std::vector<Node*>{}),
      std::invalid_argument);
  auto p1 = AtomicValue(AtomicType::PROBABILITY, 0.1);
  ConstNode cnode1(p1);
  // negative test num args can't be two
  EXPECT_THROW(
      oper::Operator(
          OperatorType::COMPLEMENT, std::vector<Node*>{&cnode1, &cnode1}),
      std::invalid_argument);
  auto r1 = AtomicValue(AtomicType::REAL, 0.1);
  ConstNode cnode2(r1);
  // negative test arg can't be real
  EXPECT_THROW(
      oper::Operator(OperatorType::COMPLEMENT, std::vector<Node*>{&cnode2}),
      std::invalid_argument);
  // complement of prob is 1-prob
  oper::Operator onode1(OperatorType::COMPLEMENT, std::vector<Node*>{&cnode1});
  EXPECT_EQ(onode1.value.type, AtomicType::PROBABILITY);
  onode1.in_nodes.push_back(&cnode1);
  std::mt19937 generator(31245);
  onode1.eval(generator);
  EXPECT_NEAR(onode1.value._double, 0.9, 0.001);
  // complement of bool is logical_not(bool)
  auto b1 = AtomicValue(false);
  ConstNode cnode3(b1);
  oper::Operator onode2(OperatorType::COMPLEMENT, std::vector<Node*>{&cnode3});
  EXPECT_EQ(onode2.value.type, AtomicType::BOOLEAN);
  onode2.in_nodes.push_back(&cnode3);
  onode2.eval(generator);
  EXPECT_EQ(onode2.value._bool, true);
}

TEST(testoperator, phi) {
  Graph g;
  // negative tests: exactly one real should be the input to a PHI
  EXPECT_THROW(
      g.add_operator(OperatorType::PHI, std::vector<uint>{}),
      std::invalid_argument);
  auto pos1 = g.add_constant_pos_real(1.0);
  EXPECT_THROW(
      g.add_operator(OperatorType::PHI, std::vector<uint>{pos1}),
      std::invalid_argument);
  auto real1 = g.add_constant(0.0);
  EXPECT_THROW(
      g.add_operator(OperatorType::PHI, std::vector<uint>{real1, real1}),
      std::invalid_argument);
  // y ~ Bernoulli(Phi(x^2))  and x = 0.5; note: Phi(0.25) = 0.5987
  auto prior = g.add_distribution(
      DistributionType::FLAT, AtomicType::REAL, std::vector<uint>{});
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{prior});
  auto x_sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{x, x});
  auto phi_x_sq = g.add_operator(OperatorType::PHI, std::vector<uint>{x_sq});
  auto likelihood = g.add_distribution(
      DistributionType::BERNOULLI,
      AtomicType::BOOLEAN,
      std::vector<uint>{phi_x_sq});
  auto y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{likelihood});
  g.query(y);
  g.observe(x, 0.5);
  const auto& means = g.infer_mean(10000, InferenceType::GIBBS);
  EXPECT_NEAR(means[0], 0.5987, 0.01);
  g.observe(y, true);
  // check gradient:
  // f(x) = log(Phi(x)); want to check f'(x) and f''(x)
  // Verified in pytorch using the following code:
  // x = torch.tensor([0.5], requires_grad=True)
  // f_x = torch.log(torch.distributions.normal.Normal(0,1).cdf(x**2))
  // f_grad = torch.autograd.grad(f_x, x, create_graph=True)
  // f_grad2 = torch.autograd.grad(f_grad, x)
  // f_grad -> 0.6458 and f_grad2 -> 0.7131
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(x, grad1, grad2);
  EXPECT_NEAR(grad1, 0.6458, 1e-3);
  EXPECT_NEAR(grad2, 0.7131, 1e-3);
}

TEST(testoperator, logistic) {
  Graph g;
  // negative tests: exactly one real should be the input to a LOGISTIC
  EXPECT_THROW(
      g.add_operator(OperatorType::LOGISTIC, std::vector<uint>{}),
      std::invalid_argument);
  auto pos1 = g.add_constant_pos_real(1.0);
  EXPECT_THROW(
      g.add_operator(OperatorType::LOGISTIC, std::vector<uint>{pos1}),
      std::invalid_argument);
  auto real1 = g.add_constant(0.0);
  EXPECT_THROW(
      g.add_operator(OperatorType::LOGISTIC, std::vector<uint>{real1, real1}),
      std::invalid_argument);
  // y ~ Bernoulli(logistic(x^2)) and x = 0.5; note: logistic(0.25) = 0.5622
  auto prior = g.add_distribution(
      DistributionType::FLAT, AtomicType::REAL, std::vector<uint>{});
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{prior});
  auto x_sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{x, x});
  auto logistic_x_sq =
      g.add_operator(OperatorType::LOGISTIC, std::vector<uint>{x_sq});
  auto likelihood = g.add_distribution(
      DistributionType::BERNOULLI,
      AtomicType::BOOLEAN,
      std::vector<uint>{logistic_x_sq});
  auto y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{likelihood});
  g.query(y);
  g.observe(x, 0.5);
  const auto& means = g.infer_mean(10000, InferenceType::GIBBS);
  EXPECT_NEAR(means[0], 0.5622, 0.01);
  g.observe(y, true);
  // check gradient:
  // f(x) = log(logistic(x^2)); want to check f'(x) and f''(x) for x=0.5
  // Verified in pytorch using the following code:
  // x = torch.tensor([0.5], requires_grad=True)
  // f_x = -torch.log(1 + torch.exp(-x**2))
  // f_grad = torch.autograd.grad(f_x, x, create_graph=True)
  // f_grad2 = torch.autograd.grad(f_grad, x)
  // f_grad -> 0.4378 and f_grad2 -> 0.6295
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(x, grad1, grad2);
  EXPECT_NEAR(grad1, 0.4378, 1e-3);
  EXPECT_NEAR(grad2, 0.6295, 1e-3);
}

TEST(testoperator, if_then_else) {
  Graph g;
  auto bool1 = g.add_constant(true);
  auto real1 = g.add_constant(10.0);
  auto real2 = g.add_constant(100.0);
  // negative tests: arg1.type==bool, arg2.type == arg3.type
  EXPECT_THROW(
      g.add_operator(OperatorType::IF_THEN_ELSE, std::vector<uint>{}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_operator(
          OperatorType::IF_THEN_ELSE, std::vector<uint>{bool1, real1, bool1}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_operator(
          OperatorType::IF_THEN_ELSE, std::vector<uint>{real1, real1, real2}),
      std::invalid_argument);
  // check eval
  auto prob1 = g.add_constant_probability(0.3);
  auto dist1 = g.add_distribution(
      DistributionType::BERNOULLI,
      AtomicType::BOOLEAN,
      std::vector<uint>{prob1});
  auto bool2 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist1});
  auto real3 = g.add_operator(
      OperatorType::IF_THEN_ELSE, std::vector<uint>{bool2, real1, real2});
  g.query(real3);
  const auto& means = g.infer_mean(10000, InferenceType::REJECTION);
  EXPECT_NEAR(means[0], 0.3 * 10 + 0.7 * 100, 1);
  // check gradient
  // logprob(x) = f(x) = real1 * y + real2 * z
  // Now y = if_then_else(bool1, x^2, x^3) and z = if_then_else(bool3, x^2, x^3)
  // since real1=10, real2=100, bool1 = true and bool3 = false we have
  // f(x) = 10 x^2 + 100 x^3
  // f'(x) = 20 x + 300 x^2 ; f'(2) = 1240
  // f''(x) = 20 + 600 x ; f''(2) = 1220
  auto prior = g.add_distribution(
      DistributionType::FLAT, AtomicType::REAL, std::vector<uint>{});
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{prior});
  g.observe(x, 2.0);
  auto x_sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{x, x});
  auto x_cube =
      g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{x, x, x});
  auto y = g.add_operator(
      OperatorType::IF_THEN_ELSE, std::vector<uint>{bool1, x_sq, x_cube});
  auto bool3 = g.add_constant(false);
  auto z = g.add_operator(
      OperatorType::IF_THEN_ELSE, std::vector<uint>{bool3, x_sq, x_cube});
  g.add_factor(FactorType::EXP_PRODUCT, std::vector<uint>{real1, y});
  g.add_factor(FactorType::EXP_PRODUCT, std::vector<uint>{real2, z});
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(x, grad1, grad2);
  EXPECT_NEAR(grad1, 1240, 1e-3);
  EXPECT_NEAR(grad2, 1220, 1e-3);
}

TEST(testoperator, log1pexp) {
  Graph g;
  // negative tests: exactly one real/pos/tensor should be the input
  EXPECT_THROW(
      g.add_operator(OperatorType::LOG1PEXP, std::vector<uint>{}),
      std::invalid_argument);
  auto prob1 = g.add_constant_probability(0.5);
  EXPECT_THROW(
      g.add_operator(OperatorType::LOG1PEXP, std::vector<uint>{prob1}),
      std::invalid_argument);
  auto pos1 = g.add_constant_pos_real(1.0);
  EXPECT_THROW(
      g.add_operator(OperatorType::LOG1PEXP, std::vector<uint>{pos1, pos1}),
      std::invalid_argument);
  // y ~ Normal(log1pexp(x^2), 1) and x = 0.5; note: ln[1+exp(0.25)] = 0.826
  auto prior = g.add_distribution(
      DistributionType::FLAT, AtomicType::REAL, std::vector<uint>{});
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{prior});
  auto x_sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{x, x});
  auto log1pexp_x_sq =
      g.add_operator(OperatorType::LOG1PEXP, std::vector<uint>{x_sq});
  auto log1pexp_x_sq_real =
      g.add_operator(OperatorType::TO_REAL, std::vector<uint>{log1pexp_x_sq});
  auto likelihood = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{log1pexp_x_sq_real, pos1});
  auto y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{likelihood});
  g.query(y);
  g.observe(x, 0.5);
  const auto& means = g.infer_mean(10000, InferenceType::NMC);
  EXPECT_NEAR(means[0], 0.826, 0.01);
  g.observe(y, 0.0);
  // check gradient:
  // Verified in pytorch using the following code:
  // x = tensor([0.5], requires_grad=True)
  // x_sq = x * x
  // f_x = dist.Normal(x_sq.exp().log1p(), tensor(1.0)).log_prob(tensor(0.0))
  // f_grad = torch.autograd.grad(f_x, x, create_graph=True)
  // f_grad2 = torch.autograd.grad(f_grad, x)
  // f_grad -> -0.4643 and f_grad2 -> -1.4480
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(x, grad1, grad2);
  EXPECT_NEAR(grad1, -0.4643, 1e-3);
  EXPECT_NEAR(grad2, -1.4480, 1e-3);
}

TEST(testoperator, logsumexp) {
  Graph g;
  // negative tests: two or more real/pos should be the input
  EXPECT_THROW(
      g.add_operator(OperatorType::LOGSUMEXP, std::vector<uint>{}),
      std::invalid_argument);
  auto pos1 = g.add_constant_pos_real(1.0);
  EXPECT_THROW(
      g.add_operator(OperatorType::LOGSUMEXP, std::vector<uint>{pos1}),
      std::invalid_argument);
  auto prob1 = g.add_constant_probability(0.5);
  EXPECT_THROW(
      g.add_operator(OperatorType::LOGSUMEXP, std::vector<uint>{prob1, prob1}),
      std::invalid_argument);
  // y ~ Normal(logsumexp(x^2, z^3), 1) and x = 0.5, z = -0.5
  // note: ln[exp(0.25) + exp(-0.125)] = 0.773
  auto prior = g.add_distribution(
      DistributionType::FLAT, AtomicType::REAL, std::vector<uint>{});
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{prior});
  auto z = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{prior});
  auto x_sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{x, x});
  auto z_thrd =
      g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{z, z, z});
  auto logsumexp_xz =
      g.add_operator(OperatorType::LOGSUMEXP, std::vector<uint>{x_sq, z_thrd});
  auto likelihood = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{logsumexp_xz, pos1});
  auto y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{likelihood});
  g.query(y);
  g.observe(x, 0.5);
  g.observe(z, -0.5);
  const auto& means = g.infer_mean(10000, InferenceType::NMC);
  EXPECT_NEAR(means[0], 0.773, 0.01);
  g.observe(y, 0.0);
  // check gradient:
  // Verified in pytorch using the following code:
  // x = tensor([0.5], requires_grad=True)
  // z = tensor([-0.5], requires_grad=True)
  // x_sq = x * x
  // z_thrd = z * z * z
  // in_nodes = torch.cat((x_sq, z_thrd), 0)
  // f_xz = dist.Normal(
  //   in_nodes.logsumexp(dim=0),
  //   tensor(1.0)).log_prob(tensor(0.0))
  // f_grad = torch.autograd.grad(f_xz, x, create_graph=True) # -0.4582
  // f_grad2 = torch.autograd.grad(f_grad, x) # -1.4543
  // f_grad = torch.autograd.grad(f_xz, z, create_graph=True) # -0.2362
  // f_grad2 = torch.autograd.grad(f_grad, z) # 0.7464
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(x, grad1, grad2);
  EXPECT_NEAR(grad1, -0.4582, 1e-3);
  EXPECT_NEAR(grad2, -1.4543, 1e-3);
  grad1 = 0;
  grad2 = 0;
  g.gradient_log_prob(z, grad1, grad2);
  EXPECT_NEAR(grad1, -0.2362, 1e-3);
  EXPECT_NEAR(grad2, 0.7464, 1e-3);
}

TEST(testoperator, log) {
  Graph g;
  // negative tests: exactly one pos_real should be the input
  EXPECT_THROW(
      g.add_operator(OperatorType::LOG, std::vector<uint>{}),
      std::invalid_argument);
  auto prob1 = g.add_constant_probability(0.5);
  EXPECT_THROW(
      g.add_operator(OperatorType::LOG, std::vector<uint>{prob1}),
      std::invalid_argument);
  auto pos1 = g.add_constant_pos_real(1.0);
  EXPECT_THROW(
      g.add_operator(OperatorType::LOG, std::vector<uint>{pos1, pos1}),
      std::invalid_argument);
  // y ~ Normal(log(x^2), 1)
  // If we observe x = 0.5 then the mean should be log(0.25) = -1.386.
  auto prior = g.add_distribution(
      DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{});
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{prior});
  auto x_sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{x, x});
  auto log_x_sq = g.add_operator(OperatorType::LOG, std::vector<uint>{x_sq});
  auto likelihood = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{log_x_sq, pos1});
  auto y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{likelihood});
  g.query(y);
  g.observe(x, 0.5);
  const auto& means = g.infer_mean(10000, InferenceType::NMC);
  EXPECT_NEAR(means[0], -1.386, 0.01);
  g.observe(y, 0.0);
  // check gradient:
  // Verified in pytorch using the following code:
  //
  // x = tensor(0.5, requires_grad=True)
  // fx = Normal((x * x).log(), tensor(1.0)).log_prob(tensor(0.0))
  // f1x = grad(fx, x, create_graph=True)
  // f2x = grad(f1x, x)
  //
  // f1x -> 5.5452 and f2x -> -27.0904
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(x, grad1, grad2);
  EXPECT_NEAR(grad1, 5.5452, 1e-3);
  EXPECT_NEAR(grad2, -27.0904, 1e-3);
}
