/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"

using namespace beanmachine::graph;

TEST(testdistrib, backward_bernoulli_noisy_or) {
  Graph g;

  uint two = g.add_constant((natural_t)2);
  uint flat_pos = g.add_distribution(
      DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{});
  uint y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_pos});

  uint dist = g.add_distribution(
      DistributionType::BERNOULLI_NOISY_OR,
      AtomicType::BOOLEAN,
      std::vector<uint>{y});
  uint x1 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist});
  uint x2 =
      g.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{dist, two});
  g.observe(y, 2.0);
  g.observe(x1, true);
  Eigen::MatrixXb x2_obs(2, 1);
  x2_obs << false, true;
  g.observe(x2, x2_obs);

  // test backward_param(), backward_value() and
  // backward_param_iid(), backward_value_iid():
  // To verify the grad1 results with pyTorch:
  // p = tensor([0.2], requires_grad=True)
  // log_prob = (
  //     dist.Binomial(10, p).log_prob(tensor(3.0)) +
  //     dist.Binomial(10, p).log_prob(tensor(4.0))
  // )
  // torch.autograd.grad(log_prob, p) -> 18.75
  EXPECT_NEAR(g.full_log_prob(), -2.2908, 1e-3);
  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 3);
  EXPECT_NEAR(grad1[0]->_double, -0.6870, 1e-3);

  // Test Bimixture
  Graph g2;
  flat_pos = g2.add_distribution(
      DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{});
  uint flat_prob = g2.add_distribution(
      DistributionType::FLAT, AtomicType::PROBABILITY, std::vector<uint>{});
  uint y1 = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_pos});
  uint y2 = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_pos});

  two = g2.add_constant((natural_t)2);
  uint d1 = g2.add_distribution(
      DistributionType::BERNOULLI_NOISY_OR,
      AtomicType::BOOLEAN,
      std::vector<uint>{y1});
  uint d2 = g2.add_distribution(
      DistributionType::BERNOULLI_NOISY_OR,
      AtomicType::BOOLEAN,
      std::vector<uint>{y2});
  uint mix_p =
      g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_prob});
  uint mix_dist = g2.add_distribution(
      DistributionType::BIMIXTURE,
      AtomicType::BOOLEAN,
      std::vector<uint>{mix_p, d1, d2});
  uint x = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{mix_dist});
  uint xiid = g2.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{mix_dist, two});
  g2.observe(y1, 0.1);
  g2.observe(y2, 2.0);
  g2.observe(mix_p, 0.7);
  g2.observe(x, false);
  Eigen::MatrixXb xiid_obs(2, 1);
  xiid_obs << false, true;
  g2.observe(xiid, xiid_obs);

  // PyTorch verification
  // p1 = torch.tensor(0.1, requires_grad=True)
  // p2 = torch.tensor(2.0, requires_grad=True)
  // p = torch.tensor(0.7, requires_grad=True)
  // x = torch.tensor([0., 0., 1.])
  // d1 = torch.distributions.Bernoulli(1 - torch.exp(-p1))
  // d2 = torch.distributions.Bernoulli(1 - torch.exp(-p2))
  // f1 = d1.log_prob(x).exp()
  // f2 = d2.log_prob(x).exp()
  // log_p = (p * f1 + (tensor(1.0) - p) * f2).log().sum()

  EXPECT_NEAR(g2.full_log_prob(), -1.9099, 1e-3);
  std::vector<DoubleMatrix*> back_grad;
  g2.eval_and_grad(back_grad);
  EXPECT_EQ(back_grad.size(), 5);
  EXPECT_NEAR(back_grad[0]->_double, 0.0633, 1e-3); // p1
  EXPECT_NEAR(back_grad[1]->_double, 0.0041, 1e-3); // p2
  EXPECT_NEAR(back_grad[2]->_double, -0.0769, 1e-3); // mix_p
}
