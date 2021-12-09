/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "beanmachine/graph/distribution/bernoulli.h"
#include "beanmachine/graph/distribution/bernoulli_noisy_or.h"
#include "beanmachine/graph/distribution/binomial.h"
#include "beanmachine/graph/distribution/tabular.h"
#include "beanmachine/graph/graph.h"

using namespace beanmachine;

#define LOG_ZERO_PT_9 ((double)-0.10536051565782628)
#define LOG_ZERO_PT_1 ((double)-2.3025850929940455)

TEST(testdistrib, bernoulli) {
  auto p1 = graph::NodeValue(graph::AtomicType::PROBABILITY, 0.1);
  graph::ConstNode cnode1(p1);
  // positive test
  distribution::Bernoulli dnode1(
      graph::AtomicType::BOOLEAN, std::vector<graph::Node*>{&cnode1});
  dnode1.in_nodes.push_back(&cnode1);
  auto zero = graph::NodeValue(false);
  auto one = graph::NodeValue(true);
  EXPECT_NEAR(LOG_ZERO_PT_9, dnode1.log_prob(zero), 1e-3);
  EXPECT_NEAR(LOG_ZERO_PT_1, dnode1.log_prob(one), 1e-3);
  // negative test for return type
  EXPECT_THROW(
      distribution::Bernoulli(
          graph::AtomicType::REAL, std::vector<graph::Node*>{&cnode1}),
      std::invalid_argument);
  // negative tests for number of arguments
  EXPECT_THROW(
      distribution::Bernoulli(
          graph::AtomicType::BOOLEAN, std::vector<graph::Node*>{}),
      std::invalid_argument);
  EXPECT_THROW(
      distribution::Bernoulli(
          graph::AtomicType::BOOLEAN,
          std::vector<graph::Node*>{&cnode1, &cnode1}),
      std::invalid_argument);
  // negative test on datatype of parents
  auto p2 = graph::NodeValue(graph::AtomicType::POS_REAL, 0.1);
  graph::ConstNode cnode2(p2);
  EXPECT_THROW(
      distribution::Bernoulli(
          graph::AtomicType::BOOLEAN, std::vector<graph::Node*>{&cnode2}),
      std::invalid_argument);

  graph::Graph g;
  auto two = g.add_constant((graph::natural_t)2);
  auto flat_dist = g.add_distribution(
      graph::DistributionType::FLAT,
      graph::AtomicType::PROBABILITY,
      std::vector<uint>{});
  auto prob =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>{flat_dist});
  auto bern_dist = g.add_distribution(
      graph::DistributionType::BERNOULLI,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>{prob});
  auto y1 =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>{bern_dist});
  auto y2 = g.add_operator(
      graph::OperatorType::IID_SAMPLE, std::vector<uint>{bern_dist, two, two});
  g.observe(prob, 0.23);
  g.observe(y1, false);
  Eigen::MatrixXb mat(2, 2);
  mat << true, true, false, true;
  g.observe(y2, mat);
  // test log_prob:
  EXPECT_NEAR(g.full_log_prob(), -4.9318, 1e-3);
  // test backward_param, backward_param_iid:
  // to verify results with pyTorch:
  // x = tensor([0.23], requires_grad=True)
  // dist_b = dist.Bernoulli(x)
  // f_x = dist_b.log_prob(True) * 3 + dist_b.log_prob(False) * 2
  // torch.autograd.grad(f_x, x)
  std::vector<graph::DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 3);
  EXPECT_NEAR(grad1[0]->_double, 10.4461, 1e-3);

  // mixture of Bernoulli-Logit
  graph::Graph g2;
  auto size = g2.add_constant((graph::natural_t)2);
  auto flat_prob = g2.add_distribution(
      graph::DistributionType::FLAT,
      graph::AtomicType::PROBABILITY,
      std::vector<uint>{});
  auto prob1 = g2.add_operator(
      graph::OperatorType::SAMPLE, std::vector<uint>{flat_prob});
  auto prob2 = g2.add_operator(
      graph::OperatorType::SAMPLE, std::vector<uint>{flat_prob});
  auto d1 = g2.add_distribution(
      graph::DistributionType::BERNOULLI,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>{prob1});
  auto d2 = g2.add_distribution(
      graph::DistributionType::BERNOULLI,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>{prob2});
  auto p = g2.add_operator(
      graph::OperatorType::SAMPLE, std::vector<uint>{flat_prob});
  auto dist = g2.add_distribution(
      graph::DistributionType::BIMIXTURE,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>{p, d1, d2});
  auto x =
      g2.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>{dist});
  auto xiid = g2.add_operator(
      graph::OperatorType::IID_SAMPLE, std::vector<uint>{dist, size});
  g2.observe(prob1, 0.2);
  g2.observe(prob2, 0.7);
  g2.observe(p, 0.65);
  g2.observe(x, true);
  Eigen::MatrixXb xobs(2, 1);
  xobs << false, true;
  g2.observe(xiid, xobs);
  // To verify the results with pyTorch:
  // p1 = torch.tensor(0.2, requires_grad=True)
  // p2 = torch.tensor(0.7, requires_grad=True)
  // p = torch.tensor(0.65, requires_grad=True)
  // x = torch.tensor([1., 0., 1.])
  // d1 = torch.distributions.Bernoulli(p1)
  // d2 = torch.distributions.Bernoulli(p2)
  // f1 = d1.log_prob(x).exp()
  // f2 = d2.log_prob(x).exp()
  // log_p = (p * f1 + (tensor(1.0) - p) * f2).log().sum()
  // torch.autograd.grad(log_p, p)[0]
  EXPECT_NEAR(g2.full_log_prob(), -2.4317, 1e-3);
  std::vector<graph::DoubleMatrix*> back_grad;
  g2.eval_and_grad(back_grad);
  EXPECT_EQ(back_grad.size(), 5);
  EXPECT_NEAR(back_grad[0]->_double, 2.4267, 1e-3); // p1
  EXPECT_NEAR(back_grad[1]->_double, 1.3067, 1e-3); // p2
  EXPECT_NEAR(back_grad[2]->_double, -1.8667, 1e-3); // p
}

TEST(testdistrib, bernoulli_noisy_or) {
  // Define log1mexp(x) = log(1 - exp(-x))
  // then log1mexp(1e-10) = -23.02585084720009
  // and log1mexp(40) = -4.248354255291589e-18
  // We will use the above facts in this test

  // first distribution
  auto p1 = graph::NodeValue(graph::AtomicType::POS_REAL, 1e-10);
  graph::ConstNode cnode1(p1);
  distribution::BernoulliNoisyOr dnode1(
      graph::AtomicType::BOOLEAN, std::vector<graph::Node*>{&cnode1});
  dnode1.in_nodes.push_back(&cnode1);
  auto zero = graph::NodeValue(false);
  auto one = graph::NodeValue(true);

  EXPECT_EQ(-1e-10, dnode1.log_prob(zero));
  EXPECT_NEAR(-23.02, dnode1.log_prob(one), 0.01);

  // second distribution
  auto p2 = graph::NodeValue(graph::AtomicType::POS_REAL, 40.0);
  graph::ConstNode cnode2(p2);
  distribution::BernoulliNoisyOr dnode2(
      graph::AtomicType::BOOLEAN, std::vector<graph::Node*>{&cnode1});
  dnode2.in_nodes.push_back(&cnode2);

  EXPECT_EQ(-40, dnode2.log_prob(zero));
  EXPECT_NEAR(-4.248e-18, dnode2.log_prob(one), 0.001e-18);
}

TEST(testdistrib, tabular) {
  Eigen::MatrixXd matrix(2, 2);
  matrix << 0.9, 0.1, 0.1, 0.9;
  graph::ConstNode cnode1(graph::NodeValue(
      graph::ValueType(
          graph::VariableType::COL_SIMPLEX_MATRIX,
          graph::AtomicType::PROBABILITY,
          matrix.rows(),
          matrix.cols()),
      matrix));
  graph::ConstNode cnode2(graph::NodeValue{true});
  distribution::Tabular dnode1(
      graph::AtomicType::BOOLEAN, std::vector<graph::Node*>{&cnode1, &cnode2});
  dnode1.in_nodes.push_back(&cnode1);
  dnode1.in_nodes.push_back(&cnode2);
  auto zero = graph::NodeValue(false);
  auto one = graph::NodeValue(true);

  EXPECT_NEAR(LOG_ZERO_PT_1, dnode1.log_prob(zero), 1e-3);
  EXPECT_NEAR(LOG_ZERO_PT_9, dnode1.log_prob(one), 1e-3);

  cnode2.value = graph::NodeValue(false);
  EXPECT_NEAR(LOG_ZERO_PT_9, dnode1.log_prob(zero), 1e-3);
  EXPECT_NEAR(LOG_ZERO_PT_1, dnode1.log_prob(one), 1e-3);
}

TEST(testdistrib, binomial) {
  auto n = graph::NodeValue((graph::natural_t)10);
  auto p = graph::NodeValue(graph::AtomicType::PROBABILITY, 0.5);
  graph::ConstNode cnode_n(n);
  graph::ConstNode cnode_p(p);
  distribution::Binomial dnode1(
      graph::AtomicType::NATURAL,
      std::vector<graph::Node*>{&cnode_n, &cnode_p});
  dnode1.in_nodes.push_back(&cnode_n);
  dnode1.in_nodes.push_back(&cnode_p);
  auto k0 = graph::NodeValue((graph::natural_t)0);
  auto k5 = graph::NodeValue((graph::natural_t)5);
  auto k11 = graph::NodeValue((graph::natural_t)11);
  EXPECT_TRUE(!std::isfinite(dnode1.log_prob(k11)));
  EXPECT_NEAR(10 * log(0.5), dnode1.log_prob(k0), 1e-2);
  // This value of -1.4020 was checked from PyTorch
  EXPECT_NEAR(-1.4020, dnode1.log_prob(k5), 1e-2);
  // negative test for return type of Binomial
  EXPECT_THROW(
      distribution::Binomial(
          graph::AtomicType::REAL,
          std::vector<graph::Node*>{&cnode_n, &cnode_p}),
      std::invalid_argument);
  // negative tests for number of arguments
  EXPECT_THROW(
      distribution::Binomial(
          graph::AtomicType::NATURAL, std::vector<graph::Node*>{&cnode_n}),
      std::invalid_argument);
  // negative test on data type of parents
  auto p2 = graph::NodeValue(graph::AtomicType::REAL, 0.5);
  graph::ConstNode cnode_p2(p2);
  EXPECT_THROW(
      distribution::Binomial(
          graph::AtomicType::NATURAL,
          std::vector<graph::Node*>{&cnode_n, &cnode_p2}),
      std::invalid_argument);
}
