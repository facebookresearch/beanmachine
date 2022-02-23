/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <stdexcept>

#include "beanmachine/graph/distribution/bernoulli.h"
#include "beanmachine/graph/distribution/beta.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/operator/unaryop.h"

using namespace beanmachine;
using namespace beanmachine::graph;
using namespace beanmachine::distribution;

TEST(testoperator, complement) {
  // negative test num args can't be zero
  EXPECT_THROW(
      oper::Complement onode1(std::vector<Node*>{}), std::invalid_argument);
  auto p1 = NodeValue(AtomicType::PROBABILITY, 0.1);
  ConstNode cnode1(p1);
  // negative test num args can't be two
  EXPECT_THROW(
      oper::Complement(std::vector<Node*>{&cnode1, &cnode1}),
      std::invalid_argument);
  auto r1 = NodeValue(AtomicType::REAL, 0.1);
  ConstNode cnode2(r1);
  // negative test arg can't be real
  EXPECT_THROW(
      oper::Complement(std::vector<Node*>{&cnode2}), std::invalid_argument);
  // complement of prob is 1-prob
  oper::Complement onode1(std::vector<Node*>{&cnode1});
  EXPECT_EQ(onode1.value.type, AtomicType::PROBABILITY);
  onode1.in_nodes.push_back(&cnode1);
  std::mt19937 generator(31245);
  onode1.eval(generator);
  EXPECT_NEAR(onode1.value._value, 0.9, 0.001);
  // complement of bool is logical_not(bool)
  auto b1 = NodeValue(false);
  ConstNode cnode3(b1);
  oper::Complement onode2(std::vector<Node*>{&cnode3});
  EXPECT_EQ(onode2.value.type, AtomicType::BOOLEAN);
  onode2.in_nodes.push_back(&cnode3);
  onode2.eval(generator);
  EXPECT_EQ(onode2.value._bool, true);
}

TEST(testoperator, multiply) {
  Graph g;
  // negative tests:
  EXPECT_THROW(
      g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{}),
      std::invalid_argument);
  auto pos1 = g.add_constant_pos_real(1.0);
  EXPECT_THROW(
      g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{pos1}),
      std::invalid_argument);
  auto real1 = g.add_constant(0.0);
  EXPECT_THROW(
      g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{real1, pos1}),
      std::invalid_argument);
  // x1, x2, x3 = Normal(0, 1)
  // m = x1 * x2 * x3
  // y ~ Normal(m, 1)
  auto normal_dist = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real1, pos1});
  auto x1 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{normal_dist});
  auto x2 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{normal_dist});
  auto x3 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{normal_dist});
  auto m =
      g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{x1, x2, x3});
  auto y_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, std::vector<uint>{m, pos1});
  auto y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{y_dist});
  // test eval()
  g.query(m);
  g.observe(x1, 1.3);
  g.observe(x2, -0.8);
  g.observe(x3, 2.1);
  const auto& means = g.infer_mean(1, InferenceType::NMC);
  EXPECT_NEAR(means[0], -2.184, 0.001);
  // test backward():
  // verification with PyTorch
  // X = tensor([1.3, -0.8, 2.1], requires_grad=True)
  // m = X.prod()
  // log_p = (
  //     dist.Normal(m, tensor(1.0)).log_prob(tensor(-0.9))
  //     + dist.Normal(tensor(0.0), tensor(1.0)).log_prob(X).sum()
  // )
  // torch.autograd.grad(log_p, X) -> [-3.4571,  4.3053, -3.4354]
  g.observe(y, -0.9);
  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 4);
  EXPECT_NEAR(grad1[0]->_double, -3.4571, 1e-3);
  EXPECT_NEAR(grad1[1]->_double, 4.3053, 1e-3);
  EXPECT_NEAR(grad1[2]->_double, -3.4354, 1e-3);
  // test backward() with 1 zero-valued input
  g.remove_observations();
  g.observe(x1, 1.3);
  g.observe(x2, -0.8);
  g.observe(x3, 0.0);
  g.observe(y, -0.9);
  g.eval_and_grad(grad1);
  EXPECT_NEAR(grad1[0]->_double, -1.3000, 1e-3);
  EXPECT_NEAR(grad1[1]->_double, 0.8000, 1e-3);
  EXPECT_NEAR(grad1[2]->_double, 0.9360, 1e-3);
  // test backward() with 2 zero-valued inputs
  g.remove_observations();
  g.observe(x1, 1.3);
  g.observe(x2, 0.0);
  g.observe(x3, 0.0);
  g.observe(y, -0.9);
  g.eval_and_grad(grad1);
  EXPECT_NEAR(grad1[0]->_double, -1.3000, 1e-3);
  EXPECT_NEAR(grad1[1]->_double, 0.0, 1e-3);
  EXPECT_NEAR(grad1[2]->_double, 0.0, 1e-3);
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
  // Verified in PyTorch using the following code:
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
  std::vector<DoubleMatrix*> grad;
  g.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 2);
  EXPECT_NEAR(grad[0]->_double, 0.6458, 1e-3);
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
  // Verified in PyTorch using the following code:
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
  std::vector<DoubleMatrix*> grad;
  g.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 2);
  EXPECT_NEAR(grad[0]->_double, 0.4378, 1e-3);
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
  std::vector<DoubleMatrix*> grad;
  g.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 2);
  EXPECT_NEAR(grad[1]->_double, 1240, 1e-3);
}

TEST(testoperator, choice) {
  Graph g;

  torch::Tensor matrix(3, 1);
  matrix << 0.25, 0.25, 0.5;
  uint simplex = g.add_constant_col_simplex_matrix(matrix);

  auto cat = g.add_distribution(
      DistributionType::CATEGORICAL,
      AtomicType::NATURAL,
      std::vector<uint>{simplex});
  auto natural = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{cat});

  auto boolean = g.add_constant(true);
  auto real1 = g.add_constant(10.0);
  auto real2 = g.add_constant(20.0);
  auto real3 = g.add_constant(30.0);

  // We need at least two parents:
  EXPECT_THROW(
      g.add_operator(OperatorType::CHOICE, std::vector<uint>{}),
      std::invalid_argument);

  EXPECT_THROW(
      g.add_operator(OperatorType::CHOICE, std::vector<uint>{natural}),
      std::invalid_argument);

  // First parent must be natural
  EXPECT_THROW(
      g.add_operator(
          OperatorType::CHOICE, std::vector<uint>{boolean, real1, real2}),
      std::invalid_argument);

  // Remaining parents must be of all the same type
  EXPECT_THROW(
      g.add_operator(
          OperatorType::CHOICE,
          std::vector<uint>{natural, boolean, real2, real3}),
      std::invalid_argument);

  // Evaluate it
  auto choice = g.add_operator(
      OperatorType::CHOICE, std::vector<uint>{natural, real1, real2, real3});

  g.query(choice);
  const auto& means = g.infer_mean(10000, InferenceType::REJECTION);
  double expected = 10.0 * 0.25 + 20.0 * 0.25 + 30.0 * 0.5;
  EXPECT_NEAR(means[0], expected, 1.0);

  // TODO: Test gradients
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
  // Verified in PyTorch using the following code:
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
  std::vector<DoubleMatrix*> grad;
  g.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 2);
  EXPECT_NEAR(grad[0]->_double, -0.4643, 1e-3);
}

TEST(testoperator, log1mexp) {
  Graph g;
  // negative tests: exactly one neg_real should be the input
  EXPECT_THROW(
      g.add_operator(OperatorType::LOG1MEXP, std::vector<uint>{}),
      std::invalid_argument);
  auto real1 = g.add_constant(0.5);
  EXPECT_THROW(
      g.add_operator(OperatorType::LOG1MEXP, std::vector<uint>{real1}),
      std::invalid_argument);
  auto pos1 = g.add_constant_pos_real(1.0);
  EXPECT_THROW(
      g.add_operator(OperatorType::LOG1MEXP, std::vector<uint>{pos1}),
      std::invalid_argument);
  auto neg1 = g.add_constant_neg_real(-1.0);
  EXPECT_THROW(
      g.add_operator(OperatorType::LOG1MEXP, std::vector<uint>{neg1, neg1}),
      std::invalid_argument);
  // y ~ Normal(log1mexp(-x^2), 1) and x = 0.5; note: ln[1-exp(-0.25)] = -1.5087
  auto prior = g.add_distribution(
      DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{});
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{prior});
  auto x_sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{x, x});
  auto neg_x_sq = g.add_operator(OperatorType::NEGATE, std::vector<uint>{x_sq});
  auto log1mexp_neg_x_sq =
      g.add_operator(OperatorType::LOG1MEXP, std::vector<uint>{neg_x_sq});
  auto log1mexp_neg_x_sq_real = g.add_operator(
      OperatorType::TO_REAL, std::vector<uint>{log1mexp_neg_x_sq});
  auto likelihood = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{log1mexp_neg_x_sq_real, pos1});
  auto y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{likelihood});
  g.query(y);
  g.observe(x, 0.5);
  const auto& means = g.infer_mean(10000, InferenceType::NMC);
  EXPECT_NEAR(means[0], -1.5087, 0.01);
  g.observe(y, 0.0);
  // check gradient:
  // Verified in PyTorch using the following code:
  // x = tensor([0.5], requires_grad=True)
  // neg_x_sq = -x * x
  // f_x = dist.Normal(
  //   neg_x_sq.exp().neg().log1p(),
  //   tensor(1.0)).log_prob(tensor(0.0))
  // f_grad = torch.autograd.grad(
  //   f_x, x, create_graph=True)
  // f_grad2 = torch.autograd.grad(f_grad, x)
  // f_grad -> 5.3118 and f_grad2 -> -25.7862
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(x, grad1, grad2);
  EXPECT_NEAR(grad1, 5.3118, 1e-3);
  EXPECT_NEAR(grad2, -25.7862, 1e-3);
  std::vector<DoubleMatrix*> grad;
  g.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 2);
  EXPECT_NEAR(grad[0]->_double, 5.3118, 1e-3);
}

TEST(testoperator, logsumexp) {
  Graph g;
  // negative tests: two or more real/pos/neg should be the input
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
  auto neg1 = g.add_constant_neg_real(-1.0);
  g.add_operator(OperatorType::LOGSUMEXP, std::vector<uint>{pos1, pos1});
  g.add_operator(OperatorType::LOGSUMEXP, std::vector<uint>{neg1, neg1});
  EXPECT_THROW(
      g.add_operator(OperatorType::LOGSUMEXP, std::vector<uint>{pos1, neg1}),
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
  // Verified in PyTorch using the following code:
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
  std::vector<DoubleMatrix*> grad;
  g.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 3);
  EXPECT_NEAR(grad[0]->_double, -0.4582, 1e-3);
  EXPECT_NEAR(grad[1]->_double, -0.2362, 1e-3);
}

TEST(testoperator, logsumexp_vector) {
  Graph g;
  // negative tests: two or more real/pos/neg should be the input
  EXPECT_THROW(
      g.add_operator(OperatorType::LOGSUMEXP_VECTOR, std::vector<uint>{}),
      std::invalid_argument);
  auto pos1 = g.add_constant_pos_real(1.0);
  EXPECT_THROW(
      g.add_operator(OperatorType::LOGSUMEXP_VECTOR, std::vector<uint>{pos1}),
      std::invalid_argument);
  auto prob1 = g.add_constant_probability(0.5);
  EXPECT_THROW(
      g.add_operator(
          OperatorType::LOGSUMEXP_VECTOR, std::vector<uint>{prob1, prob1}),
      std::invalid_argument);
  auto neg1 = g.add_constant_neg_real(-1.0);
  auto two = g.add_constant((natural_t)2);
  auto one = g.add_constant((natural_t)1);
  auto pospos = g.add_operator(OperatorType::TO_MATRIX, {two, one, pos1, pos1});
  auto negneg = g.add_operator(OperatorType::TO_MATRIX, {two, one, neg1, neg1});
  g.add_operator(OperatorType::LOGSUMEXP_VECTOR, std::vector<uint>{pospos});
  g.add_operator(OperatorType::LOGSUMEXP_VECTOR, std::vector<uint>{negneg});
  EXPECT_THROW(
      g.add_operator(
          OperatorType::LOGSUMEXP_VECTOR, std::vector<uint>{pos1, neg1}),
      std::invalid_argument);

  torch::Tensor m123(3, 1);
  m123 << 1.0, 2.0, 3.0;
  auto matrix123 = g.add_constant_real_matrix(m123);
  auto logsumexp123 = g.add_operator(
      OperatorType::LOGSUMEXP_VECTOR, std::vector<uint>{matrix123});
  g.query(logsumexp123);
  const auto& eval = g.infer(2, InferenceType::REJECTION);
  EXPECT_NEAR(eval[0][0]._value, 3.4076, 1e-3);

  auto prior = g.add_distribution(
      DistributionType::FLAT, AtomicType::REAL, std::vector<uint>{});
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{prior});
  auto z = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{prior});
  auto xz = g.add_operator(OperatorType::TO_MATRIX, {two, one, x, z});
  auto logsumexp_xz =
      g.add_operator(OperatorType::LOGSUMEXP_VECTOR, std::vector<uint>{xz});

  auto likelihood = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{logsumexp_xz, pos1});
  auto y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{likelihood});
  g.observe(x, 1.0);
  g.observe(z, 2.0);
  g.observe(y, 0.0);
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(x, grad1, grad2);
  EXPECT_NEAR(grad1, -0.6221, 1e-3);
  EXPECT_NEAR(grad2, -0.5271, 1e-3);
  grad1 = 0;
  grad2 = 0;
  g.gradient_log_prob(z, grad1, grad2);
  EXPECT_NEAR(grad1, -1.6911, 1e-3);
  EXPECT_NEAR(grad2, -0.9893, 1e-3);
  std::vector<DoubleMatrix*> grad;
  g.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 3);
  EXPECT_NEAR(grad[0]->_double, -0.6221, 1e-3);
  EXPECT_NEAR(grad[1]->_double, -1.6911, 1e-3);
}

TEST(testoperator, log) {
  Graph g;
  // negative tests: exactly one pos_real should be the input
  // or one probability should be the input.
  EXPECT_THROW(
      g.add_operator(OperatorType::LOG, std::vector<uint>{}),
      std::invalid_argument);
  auto prob1 = g.add_constant_probability(0.5);
  g.add_operator(OperatorType::LOG, std::vector<uint>{prob1});
  auto real1 = g.add_constant(-0.5);
  EXPECT_THROW(
      g.add_operator(OperatorType::LOG, std::vector<uint>{real1}),
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
  // Verified in PyTorch using the following code:
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
  std::vector<DoubleMatrix*> grad;
  g.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 2);
  EXPECT_NEAR(grad[0]->_double, 5.5452, 1e-3);
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
  auto pow_ = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{prior});
  auto x_pow = g.add_operator(OperatorType::POW, std::vector<uint>{x, pow_});
  auto x_pow_real =
      g.add_operator(OperatorType::TO_REAL, std::vector<uint>{x_pow});
  auto likelihood = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{x_pow_real, pos1});
  auto y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{likelihood});
  g.query(y);
  g.observe(x, 0.5);
  g.observe(pow_, 2.25);
  const auto& means = g.infer_mean(10000, InferenceType::NMC);
  EXPECT_NEAR(means[0], 0.21, 0.01);
  g.observe(y, 0.0);
  // check gradient:
  // Verified in PyTorch using the following code:
  //
  // x = tensor(0.5, requires_grad=True)
  // y = tensor(2.25, requires_grad=True)
  // f = Normal(x ** y, tensor(1.0)).log_prob(tensor(0.0))
  // f1x = grad(f, x, create_graph=True)
  // f2x = grad(f1x, x)
  // f1x -> -0.1989 and f2x -> -1.3921
  // f1y = grad(f, y, create_graph=True)
  // f2y = grad(f1y, y)
  // f1y -> 0.0306 and f2y -> -0.0425
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(x, grad1, grad2);
  EXPECT_NEAR(grad1, -0.1989, 1e-3);
  EXPECT_NEAR(grad2, -1.3921, 1e-3);
  grad1 = grad2 = 0;
  g.gradient_log_prob(pow_, grad1, grad2);
  EXPECT_NEAR(grad1, 0.0306, 1e-3);
  EXPECT_NEAR(grad2, -0.0425, 1e-3);
  std::vector<DoubleMatrix*> grad;
  g.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 3);
  EXPECT_NEAR(grad[0]->_double, -0.1989, 1e-3);
  EXPECT_NEAR(grad[1]->_double, 0.0306, 1e-3);

  // test x ^ y where x is negative and y is a constant
  // Verified in PyTorch using the following code:
  //
  // x = tensor(-0.5, requires_grad=True)
  // y = tensor(2.0, requires_grad=True)
  // f = Normal(x ** y, tensor(1.0)).log_prob(tensor(0.0))
  // f1x = grad(f, x, create_graph=True)
  // f2x = grad(f1x, x)
  // f1x -> 0.25 and f2x -> -1.5
  Graph g1;
  uint one = g1.add_constant_pos_real(1.0);
  y = g1.add_constant(2.0);
  uint flat_dist =
      g1.add_distribution(DistributionType::FLAT, AtomicType::REAL, {});
  x = g1.add_operator(OperatorType::SAMPLE, {flat_dist});
  uint mean = g1.add_operator(OperatorType::POW, std::vector<uint>{x, y});
  uint normal_dist = g1.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, std::vector<uint>{mean, one});
  uint z = g1.add_operator(OperatorType::SAMPLE, {normal_dist});
  g1.observe(z, 0.0);
  g1.observe(x, -0.5);
  grad1 = 0;
  grad2 = 0;
  g1.gradient_log_prob(x, grad1, grad2);
  EXPECT_NEAR(grad1, 0.25, 1e-3);
  EXPECT_NEAR(grad2, -1.5, 1e-3);

  // test x ^ y where x is negative and y is not a constant
  // Verified in PyTorch using the following code:
  //
  // x = tensor(-0.5, requires_grad=True)
  // y = 2 * x
  // f = Normal(x ** y, tensor(1.0)).log_prob(tensor(0.0))
  // f1x = grad(f, x, create_graph=True)
  // f2x = grad(f1x, x)
  // f1x -> nan and f2x -> nan
  Graph g2;
  one = g2.add_constant_pos_real(1.0);
  uint two = g2.add_constant(2.0);
  y = g1.add_constant(2.0);
  flat_dist = g2.add_distribution(DistributionType::FLAT, AtomicType::REAL, {});
  x = g2.add_operator(OperatorType::SAMPLE, {flat_dist});
  y = g2.add_operator(OperatorType::MULTIPLY, {x, two});
  mean = g2.add_operator(OperatorType::POW, std::vector<uint>{x, y});
  normal_dist = g2.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, std::vector<uint>{mean, one});
  z = g2.add_operator(OperatorType::SAMPLE, {normal_dist});
  g2.observe(z, 0.0);
  g2.observe(x, -0.5);
  grad1 = 0;
  grad2 = 0;
  g2.gradient_log_prob(x, grad1, grad2);
  EXPECT_TRUE(std::isnan(grad1));
  EXPECT_TRUE(std::isnan(grad2));
}

TEST(testoperator, iid_sample) {
  auto prob_value = NodeValue(AtomicType::PROBABILITY, 0.1);
  auto prob_node = ConstNode(prob_value);
  auto bern_dist =
      Bernoulli(AtomicType::BOOLEAN, std::vector<Node*>{&prob_node});
  bern_dist.in_nodes.push_back(&prob_node);
  auto int_value = NodeValue(AtomicType::NATURAL, (natural_t)2);
  auto int_node = ConstNode(int_value);
  auto one = ConstNode(NodeValue(AtomicType::NATURAL, (natural_t)1));

  // Negative test: There must be two or three parents.
  EXPECT_THROW(oper::IIdSample(std::vector<Node*>{}), std::invalid_argument);
  EXPECT_THROW(
      oper::IIdSample(std::vector<Node*>{&bern_dist}), std::invalid_argument);
  EXPECT_THROW(
      oper::IIdSample(
          std::vector<Node*>{&bern_dist, &int_node, &int_node, &int_node}),
      std::invalid_argument);
  // Negative test: The first parent must be a distribution.
  EXPECT_THROW(
      oper::IIdSample(std::vector<Node*>{&int_node, &int_node, &int_node}),
      std::invalid_argument);
  // Negative test: The second parent must be a constant natural.
  EXPECT_THROW(
      oper::IIdSample(std::vector<Node*>{&bern_dist, &bern_dist}),
      std::invalid_argument);
  EXPECT_THROW(
      oper::IIdSample(std::vector<Node*>{&int_node, &prob_node}),
      std::invalid_argument);
  // Negative test: The third parent must be a constant natural.
  EXPECT_THROW(
      oper::IIdSample(std::vector<Node*>{&bern_dist, &int_node, &bern_dist}),
      std::invalid_argument);
  EXPECT_THROW(
      oper::IIdSample(std::vector<Node*>{&int_node, &int_node, &prob_node}),
      std::invalid_argument);
  // Negative test: The product of the constant parents must be two or greater.
  EXPECT_THROW(
      oper::IIdSample(std::vector<Node*>{&bern_dist, &one}),
      std::invalid_argument);
  EXPECT_THROW(
      oper::IIdSample(std::vector<Node*>{&bern_dist, &one, &one}),
      std::invalid_argument);

  // test initialization
  auto pos_real_value = NodeValue(AtomicType::POS_REAL, 2.0);
  auto pos_real_node = ConstNode(pos_real_value);
  auto beta_dist = Beta(
      AtomicType::PROBABILITY,
      std::vector<Node*>{&pos_real_node, &pos_real_node});
  beta_dist.in_nodes.push_back(&pos_real_node);
  beta_dist.in_nodes.push_back(&pos_real_node);
  auto beta_samples =
      oper::IIdSample(std::vector<Node*>{&beta_dist, &int_node});
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
  torch::Tensor matrix1(2, 1);
  matrix1 << 0.6, 0.5;
  auto matrix_value = NodeValue(vtype, matrix1);
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
    x0 = *(beta_samples.value._value.data());
    x1 = *(beta_samples.value._value.data() + 1);
    mean_x0 += x0 / n_samples;
    mean_x1 += x1 / n_samples;
    mean_x0sq += x0 * x0 / n_samples;
    mean_x1sq += x1 * x1 / n_samples;
  }
  EXPECT_NEAR(mean_x0, 0.5, 0.01);
  EXPECT_NEAR(mean_x1, 0.5, 0.01);
  EXPECT_NEAR(mean_x0sq - mean_x0 * mean_x0, 1.0 / 20.0, 0.001);
  EXPECT_NEAR(mean_x1sq - mean_x1 * mean_x1, 1.0 / 20.0, 0.001);

  torch::Tensor m0 = torch::Tensor::Zero();
  for (uint i = 0; i < n_samples; i++) {
    bernoulli_samples.eval(generator);
    m0 = m0 + bernoulli_samples.value._value.cast<int>();
  }
  EXPECT_NEAR(m0.coeff(0, 0) / (double)n_samples, 0.1, 0.01);
  EXPECT_NEAR(m0.coeff(0, 1) / (double)n_samples, 0.1, 0.01);
  EXPECT_NEAR(m0.coeff(1, 0) / (double)n_samples, 0.1, 0.01);
  EXPECT_NEAR(m0.coeff(1, 1) / (double)n_samples, 0.1, 0.01);
  // log_prob_grad to be tested in each distribution test
}

TEST(testoperator, matrix_multiply) {
  Graph g;
  // negative tests:
  // requires two parents
  EXPECT_THROW(
      g.add_operator(OperatorType::MATRIX_MULTIPLY, std::vector<uint>{}),
      std::invalid_argument);
  torch::Tensor m0(2, 2);
  m0 << 0.3, -0.1, 1.2, 0.9;
  auto cm0 = g.add_constant_real_matrix(m0);
  EXPECT_THROW(
      g.add_operator(OperatorType::MATRIX_MULTIPLY, std::vector<uint>{cm0}),
      std::invalid_argument);
  // requires matrix parents
  auto c1 = g.add_constant(1.5);
  EXPECT_THROW(
      g.add_operator(OperatorType::MATRIX_MULTIPLY, std::vector<uint>{c1, c1}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_operator(OperatorType::MATRIX_MULTIPLY, std::vector<uint>{cm0, c1}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_operator(OperatorType::MATRIX_MULTIPLY, std::vector<uint>{c1, cm0}),
      std::invalid_argument);
  torch::Tensor m1(3, 2);
  m1 << 0.3, -0.1, 1.2, 0.9, -2.6, 0.8;
  auto cm1 = g.add_constant_real_matrix(m1);
  torch::Tensor m2 = torch::Tensor::Random(1, 2);
  auto cm2 = g.add_constant_bool_matrix(m2);
  // requires real/pos_real/neg_real/probability types
  EXPECT_THROW(
      g.add_operator(
          OperatorType::MATRIX_MULTIPLY, std::vector<uint>{cm2, cm0}),
      std::invalid_argument);
  // requires compatible dimension
  EXPECT_THROW(
      g.add_operator(
          OperatorType::MATRIX_MULTIPLY, std::vector<uint>{cm1, cm1}),
      std::invalid_argument);

  // test eval()
  auto zero = g.add_constant(0.0);
  auto pos1 = g.add_constant_pos_real(1.0);
  auto normal_dist = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{zero, pos1});
  auto one = g.add_constant((natural_t)1);
  auto two = g.add_constant((natural_t)2);
  auto three = g.add_constant((natural_t)3);

  auto x = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, one, three});
  auto y = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, three, two});
  auto z = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, two, two});
  auto w = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, two});

  torch::Tensor mx(1, 3);
  mx << 0.4, 0.1, 0.5;
  g.observe(x, mx);
  g.observe(y, m1);
  torch::Tensor mz(2, 2);
  mz << -1.1, 0.7, -0.6, 0.2;
  g.observe(z, mz);
  torch::Tensor mw(2, 1);
  mw << 2.3, -0.4;
  g.observe(w, mw);

  auto xy =
      g.add_operator(OperatorType::MATRIX_MULTIPLY, std::vector<uint>{x, y});
  g.query(xy);
  const auto& xy_eval = g.infer(1, InferenceType::NMC);
  EXPECT_EQ(xy_eval[0][0]._value.size(1), 2);
  EXPECT_EQ(xy_eval[0][0]._value.size(0), 1);
  // result should be simply mx * m1
  EXPECT_NEAR(xy_eval[0][0]._value.coeff(0), -1.0600, 0.001);
  EXPECT_NEAR(xy_eval[0][0]._value.coeff(1), 0.4500, 0.001);

  // test backward():
  auto zw =
      g.add_operator(OperatorType::MATRIX_MULTIPLY, std::vector<uint>{z, w});
  auto xyzw =
      g.add_operator(OperatorType::MATRIX_MULTIPLY, std::vector<uint>{xy, zw});
  auto xyzw_dist = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{xyzw, pos1});
  auto xyzw_sample =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{xyzw_dist});
  g.observe(xyzw_sample, 1.7);
  /*   to verify with PyTorch:
import torch
from torch import tensor
import torch.distributions as dist

X = tensor([0.4, 0.1, 0.5], requires_grad=True)
Y = tensor([[0.3, -0.1], [1.2, 0.9], [-2.6, 0.8]], requires_grad=True)
Z = tensor([[-1.1, 0.7], [-0.6, 0.2]], requires_grad=True)
W = tensor([2.3, -0.4], requires_grad=True)
def f_grad(x):
    XY = torch.mm(X.view((1, 3)), Y)
    ZW = torch.mm(Z, W.view((2,1)))
    XYZW = torch.mm(XY, ZW)
    log_p = (
        dist.Normal(XYZW, tensor(1.0)).log_prob(tensor(1.7))
        + dist.Normal(tensor(0.0), tensor(1.0)).log_prob(X).sum()
        + dist.Normal(tensor(0.0), tensor(1.0)).log_prob(Y).sum()
        + dist.Normal(tensor(0.0), tensor(1.0)).log_prob(Z).sum()
        + dist.Normal(tensor(0.0), tensor(1.0)).log_prob(W).sum()
    )
    return torch.autograd.grad(log_p, x)[0]

[f_grad(x) for x in [X,Y,Z,W]]

produces:

[tensor([ 0.0333,  2.8128, -4.3154]),
 tensor([[ 0.3987,  0.4630],
         [-1.0253, -0.8092],
         [ 3.4733, -0.3462]]),
 tensor([[ 2.6155, -0.9636],
         [-0.0434, -0.0881]]),
 tensor([-2.8570,  0.8053])]
  */
  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 5);
  // grad x
  EXPECT_NEAR(grad1[0]->_matrix.coeff(0), 0.0333, 1e-3);
  EXPECT_NEAR(grad1[0]->_matrix.coeff(1), 2.8128, 1e-3);
  EXPECT_NEAR(grad1[0]->_matrix.coeff(2), -4.3154, 1e-3);
  // grad y
  EXPECT_NEAR(grad1[1]->_matrix.coeff(0), 0.3987, 1e-3);
  EXPECT_NEAR(grad1[1]->_matrix.coeff(1), -1.0253, 1e-3);
  EXPECT_NEAR(grad1[1]->_matrix.coeff(2), 3.4733, 1e-3);
  EXPECT_NEAR(grad1[1]->_matrix.coeff(3), 0.4630, 1e-3);
  EXPECT_NEAR(grad1[1]->_matrix.coeff(4), -0.8092, 1e-3);
  EXPECT_NEAR(grad1[1]->_matrix.coeff(5), -0.3462, 1e-3);
  // grad z
  EXPECT_NEAR(grad1[2]->_matrix.coeff(0), 2.6155, 1e-3);
  EXPECT_NEAR(grad1[2]->_matrix.coeff(1), -0.0434, 1e-3);
  EXPECT_NEAR(grad1[2]->_matrix.coeff(2), -0.9636, 1e-3);
  EXPECT_NEAR(grad1[2]->_matrix.coeff(3), -0.0881, 1e-3);
  // grad w
  EXPECT_NEAR(grad1[3]->_matrix.coeff(0), -2.8570, 1e-3);
  EXPECT_NEAR(grad1[3]->_matrix.coeff(1), 0.8053, 1e-3);
}

TEST(testoperator, matrix_scale) {
  Graph g;
  // negative tests:
  // requires two parents
  EXPECT_THROW(
      g.add_operator(OperatorType::MATRIX_SCALE, std::vector<uint>{}),
      std::invalid_argument);
  auto c1 = g.add_constant(1.5);
  EXPECT_THROW(
      g.add_operator(OperatorType::MATRIX_SCALE, std::vector<uint>{c1}),
      std::invalid_argument);
  // requires one scalar and one matrix parent
  EXPECT_THROW(
      g.add_operator(OperatorType::MATRIX_SCALE, std::vector<uint>{c1, c1}),
      std::invalid_argument);
  torch::Tensor m1(3, 2);
  m1 << 0.3, -0.1, 1.2, 0.9, -2.6, 0.8;
  auto cm1 = g.add_constant_real_matrix(m1);
  EXPECT_THROW(
      g.add_operator(OperatorType::MATRIX_SCALE, std::vector<uint>{cm1, cm1}),
      std::invalid_argument);
  // TODO: At some point we may consider making the type signature symmetric
  //       but for the time being, this typing suffices.
  EXPECT_THROW(
      g.add_operator(OperatorType::MATRIX_SCALE, std::vector<uint>{cm1, c1}),
      std::invalid_argument);
  // Arguments should have same atomic type
  torch::Tensor mp1(3, 2);
  mp1 << 0.3, 0.1, 1.2, 0.9, 2.6, 0.8;
  auto cmp1 = g.add_constant_pos_matrix(mp1);
  EXPECT_THROW(
      g.add_operator(OperatorType::MATRIX_SCALE, std::vector<uint>{c1, cmp1}),
      std::invalid_argument);
  // requires real/pos_real/probability types
  auto cb2 = g.add_constant(false);
  torch::Tensor m2 = torch::Tensor::Random(1, 2);
  auto cmb2 = g.add_constant_bool_matrix(m2);
  EXPECT_THROW(
      g.add_operator(OperatorType::MATRIX_SCALE, std::vector<uint>{cb2, cmb2}),
      std::invalid_argument);

  // test eval()
  auto zero = g.add_constant(0.0);
  auto pos1 = g.add_constant_pos_real(1.0);
  auto normal_dist = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{zero, pos1});
  auto one = g.add_constant((natural_t)1);
  auto two = g.add_constant((natural_t)2);
  auto three = g.add_constant((natural_t)3);

  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{normal_dist});
  auto y = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, three, two});
  auto z = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, two, two});
  auto v = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, one, two});
  auto w = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{normal_dist, two});

  auto vx = 0.4;
  g.observe(x, vx);
  g.observe(y, m1);
  torch::Tensor mz(2, 2);
  mz << -1.1, 0.7, -0.6, 0.2;
  g.observe(z, mz);
  torch::Tensor mv(1, 2);
  mv << 2.3, -0.4;
  g.observe(v, mv);
  torch::Tensor mw(2, 1);
  mw << 2.3, -0.4;
  g.observe(w, mw);

  auto xy = g.add_operator(OperatorType::MATRIX_SCALE, std::vector<uint>{x, y});
  g.query(xy);
  const auto& xy_eval = g.infer(1, InferenceType::NMC);
  torch::Tensor mxy(3, 2);
  mxy = vx * m1;
  EXPECT_EQ(xy_eval[0][0]._value.size(0), 3);
  EXPECT_EQ(xy_eval[0][0]._value.size(1), 2);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_NEAR(xy_eval[0][0]._value.coeff(i + 3 * j), mxy(i, j), 0.001);
      ;
    }
  }

  // test backward():
  auto xv = g.add_operator(OperatorType::MATRIX_SCALE, std::vector<uint>{x, v});
  auto zw =
      g.add_operator(OperatorType::MATRIX_MULTIPLY, std::vector<uint>{z, w});
  auto xvzw =
      g.add_operator(OperatorType::MATRIX_MULTIPLY, std::vector<uint>{xv, zw});

  auto xyzw_dist = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{xvzw, pos1});
  auto xwzw_sample =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{xyzw_dist});
  g.observe(xwzw_sample, 1.7);

  auto zwxv =
      g.add_operator(OperatorType::MATRIX_MULTIPLY, std::vector<uint>{zw, xv});
  auto xvzwxv = g.add_operator(
      OperatorType::MATRIX_MULTIPLY, std::vector<uint>{xv, zwxv});
  auto xvzwxvzw = g.add_operator(
      OperatorType::MATRIX_MULTIPLY, std::vector<uint>{xvzwxv, zw});
  auto xvzwxvzw_dist = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{xvzwxvzw, pos1});
  auto xvzwxvzw_sample =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{xvzwxvzw_dist});
  g.observe(xvzwxvzw_sample, 1.7);
  // Reference values from PyTorch:
  /*
import torch
from torch import tensor
import torch.distributions as dist

X = tensor(0.4, requires_grad=True)
Y = tensor([[0.3, -0.1], [1.2, 0.9], [-2.6, 0.8]], requires_grad=True)
Z = tensor([[-1.1, 0.7], [-0.6, 0.2]], requires_grad=True)
V = tensor([[2.3, -0.4]], requires_grad=True)
W = tensor([[2.3], [-0.4]], requires_grad=True)
def f_grad(x):
    XV = X*V                         # (1,2)
    ZW = torch.mm(Z,W)               # (2,1)
    XVZW = torch.mm(XV, ZW)          # (1,1)
    ##
    ZWXV = torch.mm(ZW, XV)          # (2,2)
    XVZWXV = torch.mm(XV, ZWXV)      # (1,2)
    XVZWXVZW = torch.mm(XVZWXV, ZW)  # (1,1)
    ##
    log_p = (
        + dist.Normal(XVZW, tensor(1.0)).log_prob(tensor(1.7))
        + dist.Normal(XVZWXVZW, tensor(1.0)).log_prob(tensor(1.7))
        + dist.Normal(tensor(0.0), tensor(1.0)).log_prob(X).sum()
        + dist.Normal(tensor(0.0), tensor(1.0)).log_prob(Y).sum()
        + dist.Normal(tensor(0.0), tensor(1.0)).log_prob(Z).sum()
        + dist.Normal(tensor(0.0), tensor(1.0)).log_prob(V).sum()
        + dist.Normal(tensor(0.0), tensor(1.0)).log_prob(W).sum()
    )
    return torch.autograd.grad(log_p, x)[0]

[f_grad(x) for x in [X,Y,Z,V,W]]
  which produces
[tensor(-130.1199),
 tensor([[-0.3000,  0.1000],
         [-1.2000, -0.9000],
         [ 2.6000, -0.8000]]),
 tensor([[47.7895, -8.8199],
         [-7.5199,  1.2122]]),
 tensor([[-27.1010, -12.4859]]),
 tensor([[-22.5115],
         [ 13.9038]])]
  */

  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);

  EXPECT_EQ(grad1.size(), 7);

  // grad x
  EXPECT_NEAR(grad1[0]->_double, -130.1199, 1e-3);
  // grad y
  EXPECT_NEAR(grad1[1]->_matrix.coeff(0), -0.3, 1e-3);
  EXPECT_NEAR(grad1[1]->_matrix.coeff(1), -1.2, 1e-3);
  EXPECT_NEAR(grad1[1]->_matrix.coeff(2), 2.6, 1e-3);
  EXPECT_NEAR(grad1[1]->_matrix.coeff(3), 0.1, 1e-3);
  EXPECT_NEAR(grad1[1]->_matrix.coeff(4), -0.9, 1e-3);
  EXPECT_NEAR(grad1[1]->_matrix.coeff(5), -0.8, 1e-3);
  // grad z
  EXPECT_NEAR(grad1[2]->_matrix.coeff(0), 47.7895, 1e-3);
  EXPECT_NEAR(grad1[2]->_matrix.coeff(1), -7.5199, 1e-3);
  EXPECT_NEAR(grad1[2]->_matrix.coeff(2), -8.8199, 1e-3);
  EXPECT_NEAR(grad1[2]->_matrix.coeff(3), 1.2122, 1e-3);
  // grad v
  EXPECT_NEAR(grad1[3]->_matrix.coeff(0), -27.1010, 1e-3);
  EXPECT_NEAR(grad1[3]->_matrix.coeff(1), -12.4859, 1e-3);
  // grad w
  EXPECT_NEAR(grad1[4]->_matrix.coeff(0), -22.5115, 1e-3);
  EXPECT_NEAR(grad1[4]->_matrix.coeff(1), 13.9038, 1e-3);
}

TEST(testoperator, index) {
  Graph g;
  // negative initialization
  EXPECT_THROW(
      g.add_operator(OperatorType::INDEX, std::vector<uint>{}),
      std::invalid_argument);
  torch::Tensor m1(2, 1);
  m1 << 2.0, -1.0;
  auto cm1 = g.add_constant_real_matrix(m1);
  EXPECT_THROW(
      g.add_operator(OperatorType::INDEX, std::vector<uint>{cm1}),
      std::invalid_argument);
  // requires matrix and natural number
  uint real = g.add_constant(0.5);
  EXPECT_THROW(
      g.add_operator(OperatorType::INDEX, std::vector<uint>{cm1, real}),
      std::invalid_argument);

  uint zero = g.add_constant((natural_t)0);
  uint first_element =
      g.add_operator(OperatorType::INDEX, std::vector<uint>{cm1, zero});
  g.query(first_element);

  uint real_three = g.add_constant(3.0);
  uint times_three = g.add_operator(
      OperatorType::MULTIPLY, std::vector<uint>{first_element, real_three});
  g.query(times_three);

  const auto& xy_eval = g.infer(2, InferenceType::REJECTION);
  EXPECT_EQ(xy_eval[0][0]._value, 2.0);
  EXPECT_EQ(xy_eval[0][1]._value, 6.0);
}

TEST(testoperator, column_index) {
  Graph g;
  // negative initialization
  EXPECT_THROW(
      g.add_operator(OperatorType::COLUMN_INDEX, std::vector<uint>{}),
      std::invalid_argument);
  torch::Tensor m1(2, 1);
  m1 << 2.0, -1.0;
  auto cm1 = g.add_constant_real_matrix(m1);
  EXPECT_THROW(
      g.add_operator(OperatorType::COLUMN_INDEX, std::vector<uint>{cm1}),
      std::invalid_argument);
  // requires matrix and natural number
  uint real = g.add_constant(0.5);
  EXPECT_THROW(
      g.add_operator(OperatorType::COLUMN_INDEX, std::vector<uint>{cm1, real}),
      std::invalid_argument);

  torch::Tensor m2(2, 2);
  m2 << 1.0, 2.0, 3.0, 4.0;
  uint cm2 = g.add_constant_real_matrix(m2);
  uint zero = g.add_constant((natural_t)0);
  uint first_column =
      g.add_operator(OperatorType::COLUMN_INDEX, std::vector<uint>{cm2, zero});
  g.query(first_column);

  const auto& xy_eval = g.infer(2, InferenceType::REJECTION);
  EXPECT_EQ(xy_eval[0][0]._value(0), 1.0);
  EXPECT_EQ(xy_eval[0][0]._value(1), 3.0);
}

TEST(testoperator, to_real_matrix) {
  Graph g;
  // Requires exactly one input
  EXPECT_THROW(
      g.add_operator(OperatorType::TO_REAL_MATRIX, std::vector<uint>{}),
      std::invalid_argument);
  // requires matrix input
  uint three = g.add_constant_pos_real(3.0);
  EXPECT_THROW(
      g.add_operator(OperatorType::TO_REAL_MATRIX, std::vector<uint>{three}),
      std::invalid_argument);

  // Let's get a matrix of probabilities

  uint beta = g.add_distribution(
      DistributionType::BETA,
      AtomicType::PROBABILITY,
      std::vector<uint>{three, three});
  uint b1 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{beta});
  uint b2 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{beta});
  uint b3 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{beta});
  uint b4 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{beta});
  uint two = g.add_constant((natural_t)2);
  uint tm = g.add_operator(
      OperatorType::TO_MATRIX, std::vector<uint>{two, two, b1, b2, b3, b4});

  // Requires exactly one input
  EXPECT_THROW(
      g.add_operator(OperatorType::TO_REAL_MATRIX, std::vector<uint>{tm, tm}),
      std::invalid_argument);

  g.add_operator(OperatorType::TO_REAL_MATRIX, std::vector<uint>{tm});
}

TEST(testoperator, to_pos_real_matrix) {
  Graph g;
  // Requires exactly one input
  EXPECT_THROW(
      g.add_operator(OperatorType::TO_POS_REAL_MATRIX, std::vector<uint>{}),
      std::invalid_argument);
  // requires matrix input
  uint three = g.add_constant_pos_real(3.0);
  EXPECT_THROW(
      g.add_operator(
          OperatorType::TO_POS_REAL_MATRIX, std::vector<uint>{three}),
      std::invalid_argument);

  // Let's get a matrix of probabilities
  uint beta = g.add_distribution(
      DistributionType::BETA,
      AtomicType::PROBABILITY,
      std::vector<uint>{three, three});
  uint b1 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{beta});
  uint b2 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{beta});
  uint b3 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{beta});
  uint b4 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{beta});
  uint two = g.add_constant((natural_t)2);
  uint tm = g.add_operator(
      OperatorType::TO_MATRIX, std::vector<uint>{two, two, b1, b2, b3, b4});

  // Requires exactly one input
  EXPECT_THROW(
      g.add_operator(
          OperatorType::TO_POS_REAL_MATRIX, std::vector<uint>{tm, tm}),
      std::invalid_argument);

  uint tr = g.add_operator(OperatorType::TO_REAL_MATRIX, std::vector<uint>{tm});

  // Input must not be real matrix
  EXPECT_THROW(
      g.add_operator(OperatorType::TO_POS_REAL_MATRIX, std::vector<uint>{tr}),
      std::invalid_argument);

  g.add_operator(OperatorType::TO_POS_REAL_MATRIX, std::vector<uint>{tm});
}

TEST(testoperator, to_matrix) {
  Graph g;
  uint nat_one = g.add_constant((natural_t)1);
  uint nat_two = g.add_constant((natural_t)2);

  // negative initialization
  EXPECT_THROW(
      g.add_operator(OperatorType::TO_MATRIX, std::vector<uint>{}),
      std::invalid_argument);
  // requires scalar parents
  uint three = g.add_constant(3.0);
  torch::Tensor m1(2, 1);
  m1 << 2.0, -1.0;
  uint cm1 = g.add_constant_real_matrix(m1);
  EXPECT_THROW(
      g.add_operator(
          OperatorType::TO_MATRIX,
          std::vector<uint>{nat_one, nat_two, three, cm1}),
      std::invalid_argument);
  // requires parents to have the same type
  EXPECT_THROW(
      g.add_operator(
          OperatorType::TO_MATRIX,
          std::vector<uint>{nat_one, nat_two, three, nat_two}),
      std::invalid_argument);
  // incorrect row col size
  EXPECT_THROW(
      g.add_operator(
          OperatorType::TO_MATRIX,
          std::vector<uint>{nat_two, nat_two, three, three}),
      std::invalid_argument);
  // stochastic rows
  uint prob = g.add_constant_probability(0.5);
  uint bin_dist = g.add_distribution(
      DistributionType::BINOMIAL, AtomicType::NATURAL, {nat_two, prob});
  uint bin_sample = g.add_operator(OperatorType::SAMPLE, {bin_dist});
  EXPECT_THROW(
      g.add_operator(
          OperatorType::TO_MATRIX,
          std::vector<uint>{bin_sample, nat_two, three, three}),
      std::invalid_argument);

  // 1x2 real numbers
  uint two = g.add_constant(2.0);
  uint real_matrix =
      g.add_operator(OperatorType::TO_MATRIX, {nat_one, nat_two, two, three});
  g.query(real_matrix);
  const auto& eval = g.infer(2, InferenceType::REJECTION);
  EXPECT_EQ(eval[0][0]._value(0, 0), 2.0);
  EXPECT_EQ(eval[0][0]._value(0, 1), 3.0);

  // 2x2 natural numbers
  Graph g1;
  nat_two = g1.add_constant((natural_t)2);
  uint nat_zero = g1.add_constant((natural_t)0);
  uint nat_three = g1.add_constant((natural_t)3);
  uint nat_four = g1.add_constant((natural_t)4);
  uint nat_matrix = g1.add_operator(
      OperatorType::TO_MATRIX,
      {nat_two, nat_two, nat_two, nat_zero, nat_three, nat_four});
  g1.query(nat_matrix);
  const auto& eval1 = g1.infer(2, InferenceType::REJECTION);
  EXPECT_EQ(eval1[0][0]._value(0, 0), 2);
  EXPECT_EQ(eval1[0][0]._value(1, 0), 0);
  EXPECT_EQ(eval1[0][0]._value(0, 1), 3);
  EXPECT_EQ(eval1[0][0]._value(1, 1), 4);

  // 3x1 stochastic boolean samples
  Graph g2;
  nat_three = g2.add_constant((natural_t)3);
  nat_one = g2.add_constant((natural_t)1);
  uint half = g2.add_constant_probability(0.5);
  uint bern_dist = g2.add_distribution(
      DistributionType::BERNOULLI, AtomicType::BOOLEAN, {half});
  uint bern1 = g2.add_operator(OperatorType::SAMPLE, {bern_dist});
  g2.observe(bern1, false);
  uint bern2 = g2.add_operator(OperatorType::SAMPLE, {bern_dist});
  g2.observe(bern2, true);
  uint bern3 = g2.add_operator(OperatorType::SAMPLE, {bern_dist});
  g2.observe(bern3, false);
  uint bool_matrix = g2.add_operator(
      OperatorType::TO_MATRIX, {nat_three, nat_one, bern1, bern2, bern3});
  g2.query(bool_matrix);
  const auto& eval2 = g2.infer(2, InferenceType::REJECTION);
  EXPECT_EQ(eval2[0][0]._value.coeff(0), false);
  EXPECT_EQ(eval2[0][0]._value.coeff(1), true);
  EXPECT_EQ(eval2[0][0]._value.coeff(2), false);
}

TEST(testoperator, broadcast_add) {
  Graph g1;
  auto c1 = g1.add_constant(1.5);
  auto bool1 = g1.add_constant(true);
  torch::Tensor m1(2, 1);
  m1 << 2.0, -1.0;
  auto matrix1 = g1.add_constant_real_matrix(m1);
  // negative tests:
  // requires two parents
  EXPECT_THROW(
      g1.add_operator(OperatorType::BROADCAST_ADD, std::vector<uint>{}),
      std::invalid_argument);
  EXPECT_THROW(
      g1.add_operator(
          OperatorType::BROADCAST_ADD, std::vector<uint>{c1, bool1, matrix1}),
      std::invalid_argument);
  // the first parent should be a scalar and the second should be a matrix
  EXPECT_THROW(
      g1.add_operator(
          OperatorType::BROADCAST_ADD, std::vector<uint>{c1, bool1}),
      std::invalid_argument);
  EXPECT_THROW(
      g1.add_operator(
          OperatorType::BROADCAST_ADD, std::vector<uint>{matrix1, c1}),
      std::invalid_argument);
  // invalid atomic type
  EXPECT_THROW(
      g1.add_operator(
          OperatorType::BROADCAST_ADD, std::vector<uint>{bool1, matrix1}),
      std::invalid_argument);

  auto sum_matrix = g1.add_operator(
      OperatorType::BROADCAST_ADD, std::vector<uint>{c1, matrix1});
  g1.query(sum_matrix);
  const auto& eval1 = g1.infer(2, InferenceType::REJECTION);
  EXPECT_EQ(eval1[0][0]._value(0, 0), 3.5);
  EXPECT_EQ(eval1[0][0]._value(1, 0), 0.5);

  Graph g2;
  auto c2 = g2.add_constant(-1.0);
  torch::Tensor m2(2, 2);
  m2 << 0.0, 1.0, 2.0, 3.0;
  auto matrix2 = g2.add_constant_real_matrix(m2);
  auto sum_matrix2 = g2.add_operator(
      OperatorType::BROADCAST_ADD, std::vector<uint>{c2, matrix2});
  g2.query(sum_matrix2);
  const auto& eval2 = g2.infer(2, InferenceType::REJECTION);
  EXPECT_EQ(eval2[0][0]._value(0, 0), -1.0);
  EXPECT_EQ(eval2[0][0]._value(0, 1), 0.0);
  EXPECT_EQ(eval2[0][0]._value(1, 0), 1.0);
  EXPECT_EQ(eval2[0][0]._value(1, 1), 2.0);
}

TEST(testoperator, to_pos_real) {
  Graph g;
  auto zero = g.add_constant(0.0);
  auto c1 = g.add_constant(1.0);

  // normal distribution requires scale to be POS_REAL
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::NORMAL,
          AtomicType::REAL,
          std::vector<uint>{zero, c1}),
      std::invalid_argument);

  // no errors should be thrown
  auto pos_c1 =
      g.add_operator(OperatorType::TO_POS_REAL, std::vector<uint>{c1});
  auto norm_dist = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{zero, pos_c1});
  auto sample =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{norm_dist});
  g.query(sample);
  g.infer(2, InferenceType::REJECTION);

  // runtime error for negative value should be thrown
  Graph g1;
  zero = g1.add_constant(0.0);
  c1 = g1.add_constant(-1.0);
  auto invalid_c1 =
      g1.add_operator(OperatorType::TO_POS_REAL, std::vector<uint>{c1});
  norm_dist = g1.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{zero, invalid_c1});
  sample = g1.add_operator(OperatorType::SAMPLE, std::vector<uint>{norm_dist});
  g1.query(sample);
  EXPECT_THROW(g1.infer(2, InferenceType::REJECTION), std::runtime_error);
}
