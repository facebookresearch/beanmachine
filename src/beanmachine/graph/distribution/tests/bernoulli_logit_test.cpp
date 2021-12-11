/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"

using namespace beanmachine::graph;

TEST(testdistrib, bernoulli_logit) {
  Graph g;
  const double LOGIT = -1.5;
  auto logit = g.add_constant(LOGIT);
  auto pos1 = g.add_constant_pos_real(1.0);
  // negative tests: BernoulliLogit has one parent
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::BERNOULLI_LOGIT,
          AtomicType::BOOLEAN,
          std::vector<uint>{}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::BERNOULLI_LOGIT,
          AtomicType::BOOLEAN,
          std::vector<uint>{logit, logit}),
      std::invalid_argument);
  // negative test the parents must be real
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::BERNOULLI_LOGIT,
          AtomicType::BOOLEAN,
          std::vector<uint>{pos1}),
      std::invalid_argument);
  // negative test: the sample type must be BOOLEAN
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::BERNOULLI_LOGIT,
          AtomicType::REAL,
          std::vector<uint>{logit}),
      std::invalid_argument);
  // test creation of a distribution
  auto dist1 = g.add_distribution(
      DistributionType::BERNOULLI_LOGIT,
      AtomicType::BOOLEAN,
      std::vector<uint>{logit});
  // test distribution of mean and variance
  auto bool_val =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist1});
  g.query(bool_val);
  const std::vector<double>& means =
      g.infer_mean(100000, InferenceType::REJECTION);
  double prob = 1 / (1 + std::exp(-LOGIT));
  EXPECT_NEAR(means[0], prob, 0.01);
  // test log_prob
  // torch.distributions.Bernoulli(logits=-1.5).log_prob(True) -> -1.7014
  // torch.distributions.Bernoulli(logits=-1.5).log_prob(True) -> -0.2014
  g.observe(bool_val, true);
  EXPECT_NEAR(g.log_prob(bool_val), -1.7014, 0.001);
  auto bool_val2 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist1});
  g.observe(bool_val2, false);
  EXPECT_NEAR(g.log_prob(bool_val2), -0.2014, 0.001);
  // test gradient of the sampled value
  // this formula is quite straight-forward and it is not that relevant since
  // inference on discrete variables doesn't compute gradients!
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(bool_val, grad1, grad2);
  EXPECT_NEAR(grad1, LOGIT, 0.001);
  EXPECT_NEAR(grad2, 0.0, 0.001);
  // test gradient of the parameter logit
  // Verified in pytorch using the following code:
  // x = torch.tensor([1.7], requires_grad=True)
  // dist = torch.distributions.Bernoulli(logits=x**2)
  // f_x = dist.log_prob(True) + dist.log_prob(False)
  // f_grad = torch.autograd.grad(f_x, x, create_graph=True)
  // f_grad2 = torch.autograd.grad(f_grad, x)
  // f_grad -> -3.0420 and f_grad2 -> -2.9426
  logit = g.add_operator(
      OperatorType::SAMPLE,
      std::vector<uint>{g.add_distribution(
          DistributionType::FLAT, AtomicType::REAL, std::vector<uint>{})});
  auto logit_sq =
      g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{logit, logit});
  auto dist2 = g.add_distribution(
      DistributionType::BERNOULLI_LOGIT,
      AtomicType::BOOLEAN,
      std::vector<uint>{logit_sq});
  auto var1 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist2});
  auto var2 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist2});
  g.observe(var1, true);
  g.observe(var2, false);
  g.observe(logit, 1.7);
  grad1 = 0;
  grad2 = 0;
  g.gradient_log_prob(logit, grad1, grad2);
  EXPECT_NEAR(grad1, -3.0420, 0.001);
  EXPECT_NEAR(grad2, -2.9426, 0.001);
  std::vector<DoubleMatrix*> grad;
  g.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 5);
  EXPECT_NEAR(grad[2]->_double, -3.0420, 1e-3);

  auto two = g.add_constant((natural_t)2);
  auto var3 = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{dist2, two, two});
  Eigen::MatrixXb m1(2, 2);
  m1 << true, false, false, false;
  g.observe(var3, m1);
  EXPECT_NEAR(g.log_prob(logit), -11.8845, 1e-3);
  g.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 6);
  EXPECT_NEAR(grad[2]->_double, -12.5259, 1e-3);

  // mixture of Bernoulli-Logit
  Graph g2;
  auto size = g2.add_constant((natural_t)2);
  auto flat_real = g2.add_distribution(
      DistributionType::FLAT, AtomicType::REAL, std::vector<uint>{});
  auto flat_prob = g2.add_distribution(
      DistributionType::FLAT, AtomicType::PROBABILITY, std::vector<uint>{});
  auto l1 = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_real});
  auto l2 = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_real});
  auto d1 = g2.add_distribution(
      DistributionType::BERNOULLI_LOGIT,
      AtomicType::BOOLEAN,
      std::vector<uint>{l1});
  auto d2 = g2.add_distribution(
      DistributionType::BERNOULLI_LOGIT,
      AtomicType::BOOLEAN,
      std::vector<uint>{l2});
  auto p = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_prob});
  auto dist = g2.add_distribution(
      DistributionType::BIMIXTURE,
      AtomicType::BOOLEAN,
      std::vector<uint>{p, d1, d2});
  auto x = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist});
  auto xiid =
      g2.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{dist, size});
  g2.observe(l1, 1.2);
  g2.observe(l2, -1.3);
  g2.observe(p, 0.65);
  g2.observe(x, true);
  Eigen::MatrixXb xobs(2, 1);
  xobs << false, true;
  g2.observe(xiid, xobs);
  // To verify the results with pyTorch:
  // l1 = torch.tensor(1.2, requires_grad=True)
  // l2 = torch.tensor(-1.3, requires_grad=True)
  // p = torch.tensor(0.65, requires_grad=True)
  // x = torch.tensor([1., 0., 1.])
  // d1 = torch.distributions.Bernoulli(logits=l1)
  // d2 = torch.distributions.Bernoulli(logits=l2)
  // f1 = d1.log_prob(x).exp()
  // f2 = d2.log_prob(x).exp()
  // log_p = (p * f1 + (tensor(1.0) - p) * f2).log().sum()
  // torch.autograd.grad(log_p, l1)[0]
  EXPECT_NEAR(g2.full_log_prob(), -1.9630, 1e-3);
  std::vector<DoubleMatrix*> back_grad;
  g2.eval_and_grad(back_grad);
  EXPECT_EQ(back_grad.size(), 5);
  EXPECT_NEAR(back_grad[0]->_double, 0.1308, 1e-3); // l1
  EXPECT_NEAR(back_grad[1]->_double, 0.0666, 1e-3); // l2
  EXPECT_NEAR(back_grad[2]->_double, 0.6271, 1e-3); // p
}
