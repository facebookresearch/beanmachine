// Copyright (c) Facebook, Inc. and its affiliates.
#include <gtest/gtest.h>

#include "beanmachine/graph/distribution/bernoulli.h"
#include "beanmachine/graph/distribution/beta.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/operator/unaryop.h"
#include "beanmachine/graph/operator/linalgop.h"

using namespace beanmachine;
using namespace beanmachine::graph;
using namespace beanmachine::distribution;

TEST(testoperator, complement) {
  // negative test num args can't be zero
  EXPECT_THROW(
      oper::Complement onode1(std::vector<Node*>{}),
      std::invalid_argument);
  auto p1 = AtomicValue(AtomicType::PROBABILITY, 0.1);
  ConstNode cnode1(p1);
  // negative test num args can't be two
  EXPECT_THROW(
      oper::Complement(std::vector<Node*>{&cnode1, &cnode1}),
      std::invalid_argument);
  auto r1 = AtomicValue(AtomicType::REAL, 0.1);
  ConstNode cnode2(r1);
  // negative test arg can't be real
  EXPECT_THROW(
      oper::Complement(std::vector<Node*>{&cnode2}),
      std::invalid_argument);
  // complement of prob is 1-prob
  oper::Complement onode1(std::vector<Node*>{&cnode1});
  EXPECT_EQ(onode1.value.type, AtomicType::PROBABILITY);
  onode1.in_nodes.push_back(&cnode1);
  std::mt19937 generator(31245);
  onode1.eval(generator);
  EXPECT_NEAR(onode1.value._double, 0.9, 0.001);
  // complement of bool is logical_not(bool)
  auto b1 = AtomicValue(false);
  ConstNode cnode3(b1);
  oper::Complement onode2(std::vector<Node*>{&cnode3});
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
  // f_grad2 = torch.autograd.grad(f_grad, x) #-1.4543
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

TEST(testoperator, negative_log) {
  Graph g;
  // negative tests: exactly one pos_real or probability should be the input
  EXPECT_THROW(
      g.add_operator(OperatorType::NEGATIVE_LOG, std::vector<uint>{}),
      std::invalid_argument);
  auto neg_real = g.add_constant(-1.5);
  EXPECT_THROW(
      g.add_operator(OperatorType::NEGATIVE_LOG, std::vector<uint>{neg_real}),
      std::invalid_argument);
  auto pos1 = g.add_constant_pos_real(1.0);
  EXPECT_THROW(
      g.add_operator(OperatorType::NEGATIVE_LOG, std::vector<uint>{pos1, pos1}),
      std::invalid_argument);
  // y ~ Normal(-log(x^2), 1)
  // If we observe x = 0.5 then the mean should be -log(0.25) = 1.386.
  auto prior = g.add_distribution(
      DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{});
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{prior});
  auto x_sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{x, x});
  auto log_x_sq =
      g.add_operator(OperatorType::NEGATIVE_LOG, std::vector<uint>{x_sq});
  auto likelihood = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{log_x_sq, pos1});
  auto y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{likelihood});
  g.query(y);
  g.observe(x, 0.5);
  const auto& means = g.infer_mean(10000, InferenceType::NMC);
  EXPECT_NEAR(means[0], 1.386, 0.01);
  g.observe(y, 0.0);
  // check gradient:
  // Verified in pytorch using the following code:
  //
  // x = tensor(0.5, requires_grad=True)
  // fx = Normal(-(x * x).log(), tensor(1.0)).log_prob(tensor(0.0))
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

TEST(testoperator, pow) {
  Graph g;
  // There must be exactly two operands.
  EXPECT_THROW(
      g.add_operator(OperatorType::POW, std::vector<uint>{}),
      std::invalid_argument);
  auto prob1 = g.add_constant_probability(0.5);
  auto pos1 = g.add_constant_pos_real(1.0);
  auto pos225 = g.add_constant_pos_real(2.25);
  EXPECT_THROW(
      g.add_operator(OperatorType::POW, std::vector<uint>{pos1}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_operator(OperatorType::POW, std::vector<uint>{pos1, pos225, pos1}),
      std::invalid_argument);
  // Base must be prob/pos/real.
  // Power must be pos/real.
  EXPECT_THROW(
      g.add_operator(OperatorType::POW, std::vector<uint>{pos1, prob1}),
      std::invalid_argument);

  // y ~ Normal(x^2.25, 1)
  // If we observe x = 0.5 then the mean should be 0.21
  auto prior = g.add_distribution(
      DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{});
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{prior});
  auto x_pow = g.add_operator(OperatorType::POW, std::vector<uint>{x, pos225});
  auto x_pow_real =
      g.add_operator(OperatorType::TO_REAL, std::vector<uint>{x_pow});
  auto likelihood = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{x_pow_real, pos1});
  auto y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{likelihood});
  g.query(y);
  g.observe(x, 0.5);
  const auto& means = g.infer_mean(10000, InferenceType::NMC);
  EXPECT_NEAR(means[0], 0.21, 0.01);
  g.observe(y, 0.0);
  // check gradient:
  // Verified in pytorch using the following code:
  //
  // x = tensor(0.5, requires_grad=True)
  // fx = Normal(x ** 2.25, tensor(1.0)).log_prob(tensor(0.0))
  // f1x = grad(fx, x, create_graph=True)
  // f2x = grad(f1x, x)
  //
  // f1x -> -0.1989 and f2x -> -1.3921
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(x, grad1, grad2);
  EXPECT_NEAR(grad1, -0.1989, 1e-3);
  EXPECT_NEAR(grad2, -1.3921, 1e-3);
}

TEST(testoperator, iid_sample) {
  auto prob_value = AtomicValue(AtomicType::PROBABILITY, 0.1);
  auto prob_node = ConstNode(prob_value);
  auto bern_dist =
      Bernoulli(AtomicType::BOOLEAN, std::vector<Node*>{&prob_node});
  bern_dist.in_nodes.push_back(&prob_node);
  auto int_value = AtomicValue(AtomicType::NATURAL, (natural_t)2);
  auto int_node = ConstNode(int_value);
  // negative tests on the number and types of parents
  EXPECT_THROW(
      oper::IIdSample(std::vector<Node*>{}),
      std::invalid_argument);
  EXPECT_THROW(
      oper::IIdSample(std::vector<Node*>{&bern_dist, &bern_dist}),
      std::invalid_argument);
  EXPECT_THROW(
      oper::IIdSample(std::vector<Node*>{&int_node, &prob_node}),
      std::invalid_argument);

  // test initialization
  auto pos_real_value = AtomicValue(AtomicType::POS_REAL, 2.0);
  auto pos_real_node = ConstNode(pos_real_value);
  auto beta_dist = Beta(
      AtomicType::PROBABILITY,
      std::vector<Node*>{&pos_real_node, &pos_real_node});
  beta_dist.in_nodes.push_back(&pos_real_node);
  beta_dist.in_nodes.push_back(&pos_real_node);
  auto beta_samples = oper::IIdSample(std::vector<Node*>{&beta_dist, &int_node});
  beta_samples.in_nodes.push_back(&beta_dist);
  beta_samples.in_nodes.push_back(&int_node);
  auto vtype =
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::PROBABILITY, 2, 1);
  EXPECT_TRUE(beta_samples.value.type == vtype);

  auto bernoulli_samples =
      oper::IIdSample(std::vector<Node*>{&bern_dist, &int_node, &int_node});
  bernoulli_samples.in_nodes.push_back(&bern_dist);
  bernoulli_samples.in_nodes.push_back(&int_node);
  bernoulli_samples.in_nodes.push_back(&int_node);
  auto vtype2 =
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::BOOLEAN, 2, 2);
  EXPECT_TRUE(bernoulli_samples.value.type == vtype2);

  // test log_prob
  Eigen::MatrixXd matrix1(2, 1);
  matrix1 << 0.6,
             0.5;
  auto matrix_value = AtomicValue(vtype, matrix1);
  beta_samples.value = matrix_value;
  EXPECT_NEAR(beta_samples.log_prob(), 0.7701, 1e-3);

  // test eval
  std::mt19937 generator(1234);
  uint n_samples = 10000;
  double x0, x1;
  double mean_x0 = 0.0, mean_x1 = 0.0;
  double mean_x0sq = 0.0, mean_x1sq = 0.0;
  for (uint i = 0; i < n_samples; i++) {
      beta_samples.eval(generator);
      x0 = *(beta_samples.value._matrix.data());
      x1 = *(beta_samples.value._matrix.data() + 1);
      mean_x0 += x0 / n_samples;
      mean_x1 += x1 / n_samples;
      mean_x0sq += x0 * x0 / n_samples;
      mean_x1sq += x1 * x1 / n_samples;
  }
  EXPECT_NEAR(mean_x0, 0.5, 0.01);
  EXPECT_NEAR(mean_x1, 0.5, 0.01);
  EXPECT_NEAR(mean_x0sq - mean_x0 * mean_x0, 1.0 / 20.0, 0.001);
  EXPECT_NEAR(mean_x1sq - mean_x1 * mean_x1, 1.0 / 20.0, 0.001);

  Eigen::Matrix2i m0 = Eigen::Matrix2i::Zero();
  for (uint i = 0; i < n_samples; i++) {
    bernoulli_samples.eval(generator);
    m0 = m0.array() + bernoulli_samples.value._bmatrix.cast<int>().array();
  }
  EXPECT_NEAR(m0.coeff(0, 0) / (double)n_samples, 0.1, 0.01);
  EXPECT_NEAR(m0.coeff(0, 1) / (double)n_samples, 0.1, 0.01);
  EXPECT_NEAR(m0.coeff(1, 0) / (double)n_samples, 0.1, 0.01);
  EXPECT_NEAR(m0.coeff(1, 1) / (double)n_samples, 0.1, 0.01);
  // log_prob_grad to be tested in each distribution test
}

TEST(testoperator, matrix_multiply) {
  Graph g;
  // negative tests: take exactly two parents
  EXPECT_THROW(
      g.add_operator(OperatorType::MATRIX_MULTIPLY, std::vector<uint>{}),
      std::invalid_argument);
  Eigen::MatrixXd m1 = Eigen::MatrixXd::Constant(2, 3, 0.5);
  auto m1_node = g.add_constant_matrix(m1);
  EXPECT_THROW(
      g.add_operator(OperatorType::MATRIX_MULTIPLY, std::vector<uint>{m1_node}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_operator(
          OperatorType::MATRIX_MULTIPLY,
          std::vector<uint>{m1_node, m1_node, m1_node}),
      std::invalid_argument);
  // negative tests: two parents must be BROADCAST_MATRIX
  auto natural_2 = g.add_constant((graph::natural_t)2);
  EXPECT_THROW(
      g.add_operator(
          OperatorType::MATRIX_MULTIPLY, std::vector<uint>{m1_node, natural_2}),
      std::invalid_argument);
  // negative tests: two parents must have the same AtomicType:
  // real, pos_real, or probability
  Eigen::MatrixXn m2 = Eigen::MatrixXn::Constant(3, 1, (graph::natural_t)2);
  auto m2_node = g.add_constant_matrix(m2);
  auto m3_node = g.add_constant_probability_matrix(m1);
  EXPECT_THROW(
      g.add_operator(
          OperatorType::MATRIX_MULTIPLY, std::vector<uint>{m1_node, m2_node}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_operator(
          OperatorType::MATRIX_MULTIPLY, std::vector<uint>{m1_node, m3_node}),
      std::invalid_argument);
  // negative tests: two parents must have compatible dimensions:
  EXPECT_THROW(
      g.add_operator(
          OperatorType::MATRIX_MULTIPLY, std::vector<uint>{m1_node, m1_node}),
      std::invalid_argument);

  // test eval
  Eigen::MatrixXd mat_a(2, 3);
  mat_a << 1.3, 0.5, 0.1,
           1.2, -0.9, -0.3;
  Eigen::MatrixXd mat_b(3, 2);
  mat_b << -3.0, 0.8,
           -2.1, -1.1,
           -1.8, -0.1;
  Eigen::MatrixXd mat_c(2, 2);
  mat_c << -5.13, 0.48,
           -1.17, 1.98;
  auto real_matrix_A = ConstNode(AtomicValue(mat_a));
  auto real_matrix_B = ConstNode(AtomicValue(mat_b));
  auto AmmB = oper::MatrixMultiply(std::vector<Node*>{&real_matrix_A, &real_matrix_B});
  AmmB.in_nodes.push_back(&real_matrix_A);
  AmmB.in_nodes.push_back(&real_matrix_B);
  std::mt19937 generator(1234);
  AmmB.eval(generator);
  EXPECT_EQ(AmmB.value.type.variable_type, graph::VariableType::BROADCAST_MATRIX);
  EXPECT_TRUE(AmmB.value._matrix.isApprox(mat_c));

  // test auto conversion to scalar
  Eigen::MatrixXd row0 = mat_a.row(0);
  Eigen::MatrixXd col0 = mat_b.col(0);
  auto row_vec = ConstNode(AtomicValue(row0));
  auto col_vec = ConstNode(AtomicValue(col0));
  auto inner_prod = oper::MatrixMultiply(std::vector<Node*>{&row_vec, &col_vec});
  inner_prod.in_nodes.push_back(&row_vec);
  inner_prod.in_nodes.push_back(&col_vec);
  inner_prod.eval(generator);
  EXPECT_EQ(inner_prod.value.type.variable_type, graph::VariableType::SCALAR);
  EXPECT_EQ(inner_prod.value._double, -5.13);

  // test gradient
  // a = torch.tensor([[0.4, 0.6]])
  // b = torch.tensor([[0.8, 0.2]])
  // x = torch.tensor([[0.8], [0.7]], requires_grad=True)
  // prior = dist.Beta(2.0, 1.0).log_prob(x).sum()
  // likelihood = dist.Beta(
  //     torch.mm(a, x).item(), torch.mm(b, x).item()).log_prob(torch.tensor(0.6))
  // f_x = likelihood + prior
  // f_grad_x = torch.autograd.grad(f_x, x, create_graph=True)[0]
  // torch.autograd.grad(f_grad_x[0][0], x)[0]
  // torch.autograd.grad(f_grad_x[1][0], x)[0]
  auto pos1 = g.add_constant_pos_real(2.0);
  auto pos2 = g.add_constant_pos_real(1.0);
  auto beta_prior = g.add_distribution(
      DistributionType::BETA, AtomicType::PROBABILITY, std::vector<uint>{pos1, pos2});
  auto x = g.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{beta_prior, natural_2});
  Eigen::MatrixXd col_x(2, 1);
  col_x << 0.8,
           0.7;
  g.observe(x, col_x);
  Eigen::MatrixXd row_a(1, 2);
  row_a << 0.4, 0.6;
  Eigen::MatrixXd row_b(1, 2);
  row_b << 0.8, 0.2;
  auto row_A = g.add_constant_probability_matrix(row_a);
  auto row_B = g.add_constant_probability_matrix(row_b);

  auto Ax = g.add_operator(OperatorType::MATRIX_MULTIPLY, std::vector<uint>{row_A, x});
  auto Bx = g.add_operator(OperatorType::MATRIX_MULTIPLY, std::vector<uint>{row_B, x});
  auto beta_likelihood = g.add_distribution(
      DistributionType::BETA, AtomicType::PROBABILITY, std::vector<uint>{Ax, Bx});
  auto y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{beta_likelihood});
  g.observe(y, 0.6);
  g.query(y);
  // eval from the src to initialize the graph
  g.log_prob(x);

  Eigen::MatrixXd Grad1 = Eigen::MatrixXd::Zero(1, 2);
  Eigen::MatrixXd Grad2 = Eigen::MatrixXd::Zero(1, 2);
  g.gradient_log_prob(x, Grad1, Grad2);
  EXPECT_NEAR(Grad1.coeff(0), 1.2500, 0.001);
  EXPECT_NEAR(Grad1.coeff(1), 1.4286, 0.001);
  EXPECT_NEAR(Grad2.coeff(0), -1.5625, 0.001);
  EXPECT_NEAR(Grad2.coeff(1), -2.0408, 0.001);
}
