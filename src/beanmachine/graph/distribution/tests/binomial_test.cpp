/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"

using namespace beanmachine::graph;

TEST(testdistrib, backward_binomial) {
  Graph g;
  uint two = g.add_constant((natural_t)2);
  uint ten = g.add_constant((natural_t)10);

  uint flat_dist = g.add_distribution(
      DistributionType::FLAT, AtomicType::PROBABILITY, std::vector<uint>{});
  uint p = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_dist});

  uint bin_dist = g.add_distribution(
      DistributionType::BINOMIAL,
      AtomicType::NATURAL,
      std::vector<uint>{ten, p});
  uint k1 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{bin_dist});
  uint k2 = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{bin_dist});
  g.observe(p, 0.2);
  g.observe(k1, (natural_t)3);
  g.observe(k2, (natural_t)4);

  // test backward_param(), backward_value() and
  // backward_param_iid(), backward_value_iid():
  // To verify the grad1 results with pyTorch:
  // p = tensor([0.2], requires_grad=True)
  // log_prob = (
  //     dist.Binomial(10, p).log_prob(tensor(3.0)) +
  //     dist.Binomial(10, p).log_prob(tensor(4.0))
  // )
  // torch.autograd.grad(log_prob, p) -> 18.75
  std::vector<DoubleMatrix*> grad1;
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 3);
  EXPECT_NEAR(grad1[0]->_double, 18.75, 1e-3);

  g.remove_observations();
  g.observe(p, 0.2);
  uint k = g.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{bin_dist, two, two});
  torch::Tensor k_obs(2, 2);
  k_obs << (natural_t)2, (natural_t)1, (natural_t)3, (natural_t)3;
  g.observe(k, k_obs);

  // To verify the grad1 results with pyTorch:
  // p = tensor([0.2], requires_grad=True)
  // k = tensor([2.0, 1.0, 3.0, 3.0])
  // log_prob = dist.Binomial(10, p).log_prob(k).sum()
  // torch.autograd.grad(log_prob, p) -> 6.25

  // test log_prob() on vector value:
  EXPECT_NEAR(g.log_prob(k), -5.7182, 1e-3);
  g.eval_and_grad(grad1);
  EXPECT_EQ(grad1.size(), 2);
  EXPECT_NEAR(grad1[0]->_double, 6.25, 1e-3);

  // Test Bimixture
  Graph g2;
  uint flat_prob = g2.add_distribution(
      DistributionType::FLAT, AtomicType::PROBABILITY, std::vector<uint>{});
  uint prob1 =
      g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_prob});
  uint prob2 =
      g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_prob});

  two = g2.add_constant((natural_t)2);
  ten = g2.add_constant((natural_t)10);
  uint d1 = g2.add_distribution(
      DistributionType::BINOMIAL,
      AtomicType::NATURAL,
      std::vector<uint>{ten, prob1});
  uint d2 = g2.add_distribution(
      DistributionType::BINOMIAL,
      AtomicType::NATURAL,
      std::vector<uint>{ten, prob2});
  uint mix_p =
      g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_prob});
  uint dist = g2.add_distribution(
      DistributionType::BIMIXTURE,
      AtomicType::NATURAL,
      std::vector<uint>{mix_p, d1, d2});
  k = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist});
  uint kiid =
      g2.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{dist, two});
  g2.observe(prob1, 0.5);
  g2.observe(prob2, 0.1);
  g2.observe(mix_p, 0.7);
  g2.observe(k, (natural_t)4);
  torch::Tensor kiid_obs(2, 1);
  kiid_obs << (natural_t)6, (natural_t)1;
  g2.observe(kiid, kiid_obs);

  // PyTorch verification
  // p1 = torch.tensor(0.5, requires_grad=True)
  // p2 = torch.tensor(0.1, requires_grad=True)
  // p = torch.tensor(0.7, requires_grad=True)
  // x = torch.tensor([4., 6., 1.])
  // d1 = torch.distributions.Binomial(10, p1)
  // d2 = torch.distributions.Binomial(10, p2)
  // f1 = d1.log_prob(x).exp()
  // f2 = d2.log_prob(x).exp()
  // log_p = (p * f1 + (tensor(1.0) - p) * f2).log().sum()
  EXPECT_NEAR(g2.full_log_prob(), -5.9538, 1e-3);
  std::vector<DoubleMatrix*> back_grad;
  g2.eval_and_grad(back_grad);
  EXPECT_EQ(back_grad.size(), 5);
  EXPECT_NEAR(back_grad[0]->_double, -0.7988, 1e-3); // p1
  EXPECT_NEAR(back_grad[1]->_double, 0.7757, 1e-3); // p2
  EXPECT_NEAR(back_grad[2]->_double, -0.3216, 1e-3); // mix_p
}
