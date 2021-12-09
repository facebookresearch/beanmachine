/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/util.h"

using namespace beanmachine;
using namespace beanmachine::graph;

TEST(testdistrib, half_cauchy) {
  Graph g;
  auto real1 = g.add_constant(4.5);
  const double SCALE = 3.5;
  auto pos1 = g.add_constant_pos_real(SCALE);
  // negative tests: half cauchy has one parent which is positive real
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::HALF_CAUCHY,
          AtomicType::POS_REAL,
          std::vector<uint>{}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::HALF_CAUCHY,
          AtomicType::POS_REAL,
          std::vector<uint>{real1}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::HALF_CAUCHY,
          AtomicType::POS_REAL,
          std::vector<uint>{pos1, pos1}),
      std::invalid_argument);
  // negative test: sample type should be positive real
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::HALF_CAUCHY,
          AtomicType::REAL,
          std::vector<uint>{pos1}),
      std::invalid_argument);
  // test creation of a distribution
  auto half_cauchy_dist = g.add_distribution(
      DistributionType::HALF_CAUCHY,
      AtomicType::POS_REAL,
      std::vector<uint>{pos1});
  // test percentiles of the sampled value from a Half Cauchy distribution
  auto pos_val =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{half_cauchy_dist});
  g.query(pos_val);
  std::vector<std::vector<NodeValue>>& samples =
      g.infer(10000, InferenceType::REJECTION);
  std::vector<double> values;
  for (const auto& sample : samples) {
    values.push_back(sample[0]._double);
  }
  auto perc_values =
      util::percentiles<double>(values, std::vector<double>{.25, .5, .75, .9});
  EXPECT_NEAR(perc_values[0], std::tan(.25 * M_PI_2) * SCALE, .1);
  EXPECT_NEAR(perc_values[1], std::tan(.50 * M_PI_2) * SCALE, .1);
  EXPECT_NEAR(perc_values[2], std::tan(.75 * M_PI_2) * SCALE, .1);
  EXPECT_NEAR(perc_values[3], std::tan(.90 * M_PI_2) * SCALE, 1);
  // test log_prob
  g.observe(pos_val, 7.0);
  EXPECT_NEAR(
      g.log_prob(pos_val), -3.3138, 0.001); // value computed from pytorch
  // test gradient of value and scale
  auto pos_sq_val = g.add_operator(
      OperatorType::MULTIPLY, std::vector<uint>{pos_val, pos_val});
  auto half_cauchy_dist2 = g.add_distribution(
      DistributionType::HALF_CAUCHY,
      AtomicType::POS_REAL,
      std::vector<uint>{pos_sq_val});
  auto pos_val2 = g.add_operator(
      OperatorType::SAMPLE, std::vector<uint>{half_cauchy_dist2});
  g.observe(pos_val2, 100.0);
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(pos_val, grad1, grad2);
  // f(x) =  -log(pi/2) +log(3.5) -log(3.5^2 + x^2) -log(pi/2) +log(x^2)
  // -log(x^4 + 100^2) f'(x) = -2x /(3.5^2 + x^2) + 2/x -4x^3/(x^4 + 100^2)
  // f'(7) = -0.053493
  // f''(x) = -2/(3.5^2 + x^2) + 4x^2 /(3.5^2 + x^2)^2 - 2/x^2 - 12x^2/(x^4 +
  // 100^2) + 16x^6/(x^4 + 100^2)^2 f''(7) = -0.056399
  EXPECT_NEAR(grad1, -0.053493, 1e-6);
  EXPECT_NEAR(grad2, -0.056399, 1e-6);
  std::vector<DoubleMatrix*> grad;
  g.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 2);
  EXPECT_NEAR(grad[0]->_double, -0.0535, 1e-3);
  EXPECT_NEAR(grad[1]->_double, -0.0161, 1e-3);

  // test vector operators
  // to verify results with pyTorch:
  // X = tensor(0.8, requires_grad=True)
  // Y = tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
  // log_p = (
  //     dist.HalfCauchy(X).log_prob(Y).sum()
  //     + dist.HalfCauchy(tensor(1.1)).log_prob(X)
  // )
  // torch.autograd.grad(log_p, X) -> -4.8711
  // torch.autograd.grad(log_p, Y) -> [[-0.3077, -0.5882], [-0.8219, -1.0000]]
  Graph g2;
  auto two = g2.add_constant((natural_t)2);
  auto pos2 = g2.add_constant_pos_real(1.1);
  auto hc_dist = g2.add_distribution(
      DistributionType::HALF_CAUCHY,
      AtomicType::POS_REAL,
      std::vector<uint>{pos2});
  auto x = g2.add_operator(OperatorType::SAMPLE, std::vector<uint>{hc_dist});
  auto y_dist = g2.add_distribution(
      DistributionType::HALF_CAUCHY,
      AtomicType::POS_REAL,
      std::vector<uint>{x});
  auto y = g2.add_operator(
      OperatorType::IID_SAMPLE, std::vector<uint>{y_dist, two, two});
  g2.observe(x, 0.8);
  Eigen::MatrixXd m_y(2, 2);
  m_y << 0.1, 0.2, 0.3, 0.4;
  g2.observe(y, m_y);
  // test log_prob():
  EXPECT_NEAR(g2.log_prob(x), -2.3161, 1e-3);
  // test backward_value/param/ (iid):
  g2.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 2);
  EXPECT_NEAR(grad[0]->_double, -4.8711, 1e-3);
  EXPECT_NEAR(grad[1]->_matrix.coeff(0), -0.3077, 1e-3);
  EXPECT_NEAR(grad[1]->_matrix.coeff(1), -0.8219, 1e-3);
  EXPECT_NEAR(grad[1]->_matrix.coeff(2), -0.5882, 1e-3);
  EXPECT_NEAR(grad[1]->_matrix.coeff(3), -1.0000, 1e-3);

  // mixture of half_cauchy
  Graph g3;
  auto size = g3.add_constant((natural_t)2);
  auto flat_pos = g3.add_distribution(
      DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{});
  auto flat_prob = g3.add_distribution(
      DistributionType::FLAT, AtomicType::PROBABILITY, std::vector<uint>{});
  auto s1 = g3.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_pos});
  auto s2 = g3.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_pos});
  auto d1 = g3.add_distribution(
      DistributionType::HALF_CAUCHY,
      AtomicType::POS_REAL,
      std::vector<uint>{s1});
  auto d2 = g3.add_distribution(
      DistributionType::HALF_CAUCHY,
      AtomicType::POS_REAL,
      std::vector<uint>{s2});
  auto p = g3.add_operator(OperatorType::SAMPLE, std::vector<uint>{flat_prob});
  auto dist = g3.add_distribution(
      DistributionType::BIMIXTURE,
      AtomicType::POS_REAL,
      std::vector<uint>{p, d1, d2});
  auto x_ = g3.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist});
  auto xiid =
      g3.add_operator(OperatorType::IID_SAMPLE, std::vector<uint>{dist, size});
  g3.observe(s1, 1.1);
  g3.observe(s2, 4.3);
  g3.observe(p, 0.65);
  g3.observe(x_, 3.5);
  Eigen::MatrixXd xobs(2, 1);
  xobs << 0.5, 1.5;
  g3.observe(xiid, xobs);
  // To verify the results with pyTorch:
  // s1 = torch.tensor(1.1, requires_grad=True)
  // s2 = torch.tensor(4.3, requires_grad=True)
  // p = torch.tensor(0.65, requires_grad=True)
  // x = torch.tensor([3.5, 0.5, 1.5], requires_grad=True)
  // d1 = torch.distributions.HalfCauchy(s1)
  // d2 = torch.distributions.HalfCauchy(s2)
  // f1 = d1.log_prob(x).exp()
  // f2 = d2.log_prob(x).exp()
  // log_p = (p * f1 + (tensor(1.0) - p) * f2).log().sum()
  // torch.autograd.grad(log_p, x)[0]
  EXPECT_NEAR(g3.full_log_prob(), -5.4746, 1e-3);
  std::vector<DoubleMatrix*> back_grad;
  g3.eval_and_grad(back_grad);
  EXPECT_EQ(back_grad.size(), 5);
  EXPECT_NEAR(back_grad[0]->_double, 0.0767, 1e-3); // s1
  EXPECT_NEAR(back_grad[1]->_double, -0.1019, 1e-3); // s2
  EXPECT_NEAR(back_grad[2]->_double, 0.7455, 1e-3); // p
  EXPECT_NEAR(back_grad[3]->_double, -0.3798, 1e-3); // x_
  EXPECT_NEAR(back_grad[4]->_matrix.coeff(0), -0.5960, 1e-3); // xiid
  EXPECT_NEAR(back_grad[4]->_matrix.coeff(1), -0.6793, 1e-3);
}
