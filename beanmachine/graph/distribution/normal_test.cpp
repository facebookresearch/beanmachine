// Copyright (c) Facebook, Inc. and its affiliates.
#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"

using namespace beanmachine::graph;

TEST(testdistrib, normal) {
  Graph g;
  const double MEAN = -11.0;
  const double STD = 3.0;
  auto real1 = g.add_constant(MEAN);
  auto pos1 = g.add_constant_pos_real(STD);
  // negative tests normal has two parents
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::NORMAL, AtomicType::REAL, std::vector<uint>{}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::NORMAL, AtomicType::REAL, std::vector<uint>{real1}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::NORMAL,
          AtomicType::REAL,
          std::vector<uint>{real1, pos1, real1}),
      std::invalid_argument);
  // negative test the parents must be a real and a positive
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::NORMAL,
          AtomicType::REAL,
          std::vector<uint>{real1, real1}),
      std::invalid_argument);
  // test creation of a distribution
  auto normal_dist = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real1, pos1});
  // test distribution of mean and variance
  auto real_val =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{normal_dist});
  auto real_sq_val = g.add_operator(
      OperatorType::MULTIPLY, std::vector<uint>{real_val, real_val});
  g.query(real_val);
  g.query(real_sq_val);
  const std::vector<double>& means =
      g.infer_mean(100000, InferenceType::REJECTION);
  EXPECT_NEAR(means[0], MEAN, 0.1);
  EXPECT_NEAR(means[1] - means[0] * means[0], STD * STD, 0.1);
  // test log_prob
  g.observe(real_val, 1.0);
  EXPECT_NEAR(
      g.log_prob(real_val), -10.0176, 0.001); // value computed from pytorch
  // test gradient of log_prob w.r.t. value and the mean
  // f(x) = -.5 (x + 11)^2 / 9  -.5 (x^2 - 3)^2/9
  // f'(x) = -(x + 11) / 9 - (x^2 - 3) (2x) / 9
  // f''(x) = -1/9 - 4x^2/9 - 2 (x^2 - 3) /9
  auto normal_dist2 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real_sq_val, pos1});
  auto real_val2 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{normal_dist2});
  g.observe(real_val2, 3.0);
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(real_val, grad1, grad2);
  EXPECT_NEAR(grad1, -0.88888, 0.01);
  EXPECT_NEAR(grad2, -0.11111, 0.01);
  // test gradient of log_prob w.r.t. sigma
  const double SCALE = 3.5;
  auto pos_scale = g.add_constant_pos_real(SCALE);
  auto half_cauchy_dist = g.add_distribution(
      DistributionType::HALF_CAUCHY,
      AtomicType::POS_REAL,
      std::vector<uint>{pos_scale});
  auto pos_val =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{half_cauchy_dist});
  g.observe(pos_val, 7.0);
  auto pos_sq_val = g.add_operator(
      OperatorType::MULTIPLY, std::vector<uint>{pos_val, pos_val});
  auto normal_dist3 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real1, pos_sq_val});
  auto real_val3 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{normal_dist3});
  g.observe(real_val3, 5.0);
  grad1 = grad2 = 0;
  g.gradient_log_prob(pos_val, grad1, grad2);
  // f(x) = -log(pi/2) + log(3.5) - log(3.5^2 + x^2) -0.5 log(2pi) -log(x^2)
  // -0.5*(5 + 11)^2/x^4 f'(x) = -2x/(3.5^2 + x^2) -2/x +0.5*16^2*4/x^5 f'(7) =
  // -0.483822 f''(x) = -2/(3.5^2 + x^2) + 4x^2/(3.5^2 + x^2)^2   +2/x^2
  // -0.5*16^2*4*5/x^6 f''(7) = 0.038648
  EXPECT_NEAR(grad1, -0.483822, 1e-6);
  EXPECT_NEAR(grad2, 0.038648, 1e-6);
}
