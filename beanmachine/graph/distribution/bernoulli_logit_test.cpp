// Copyright (c) Facebook, Inc. and its affiliates.
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
}
