// Copyright (c) Facebook, Inc. and its affiliates.
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
  std::vector<std::vector<AtomicValue>>& samples =
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
}
