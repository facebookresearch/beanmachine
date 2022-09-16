/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define _USE_MATH_DEFINES
#include <cmath>

#include <gtest/gtest.h>

#include <beanmachine/graph/operator/stochasticop.h>
#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/util.h"

using namespace beanmachine;
using namespace beanmachine::graph;
using namespace beanmachine::distribution;

TEST(testdistrib, cauchy) {
  Graph g;
  const double SCALE = 3.5;
  const double X0 = 1.2;
  auto scale_pos = g.add_constant_pos_real(SCALE);
  auto x0_real = g.add_constant_real(X0);
  auto pos1 = g.add_constant_pos_real(SCALE);
  auto real1 = g.add_constant_real(SCALE);
  // negative tests: cauchy has two parents which are real and positive
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::CAUCHY, AtomicType::REAL, std::vector<uint>{}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::CAUCHY,
          AtomicType::REAL,
          std::vector<uint>{x0_real}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::CAUCHY,
          AtomicType::REAL,
          std::vector<uint>{pos1, pos1}),
      std::invalid_argument);
  // negative test: sample type should be real
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::CAUCHY,
          AtomicType::REAL,
          std::vector<uint>{pos1, scale_pos}),
      std::invalid_argument);
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::CAUCHY,
          AtomicType::REAL,
          std::vector<uint>{x0_real, real1}),
      std::invalid_argument);

  // test creation of a distribution

  auto zero = g.add_constant_real(0.0);
  auto pos_zero = g.add_constant_pos_real(0.0);
  auto loc =
      g.add_operator(OperatorType::ADD, std::vector<uint>{zero, x0_real});
  auto scale =
      g.add_operator(OperatorType::ADD, std::vector<uint>{pos_zero, scale_pos});
  auto cauchy_dist = g.add_distribution(
      DistributionType::CAUCHY,
      AtomicType::REAL,
      std::vector<uint>{loc, scale});

  // test percentiles of the sampled value from a Cauchy distribution
  auto sample =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{cauchy_dist});
  g.query(sample);
  std::vector<std::vector<NodeValue>>& samples =
      g.infer(10000, InferenceType::REJECTION);
  std::vector<double> values;
  values.reserve(samples.size());
  for (const auto& s : samples) {
    values.push_back(s[0]._double);
  }
  auto quantiles = std::vector<double>{0.10, 0.25, 0.50, 0.75, 0.90};
  auto perc_values = util::percentiles<double>(values, quantiles);
  auto quantile = [X0, SCALE](double p) {
    return X0 + SCALE * std::tan(M_PI * (p - 0.5));
  };
  for (int i = 0; i < quantiles.size(); i++) {
    auto p = quantiles[i];
    auto q = quantile(p);
    EXPECT_NEAR(perc_values[i], q, abs(q) * 0.1);
  }

  auto x1 = 7.0;
  g.observe(sample, x1);

  // test log_prob
  {
    auto logProb = [X0, SCALE](double x) {
      return log(1 / (M_PI * SCALE * (1 + pow((x - X0) / SCALE, 2))));
    };
    auto actual = g.log_prob(sample);
    auto expected = logProb(x1);
    EXPECT_NEAR(actual, expected, 0.00001);
    EXPECT_NEAR(g.log_prob(sample), expected, 0.00001);
  }

  auto cauchy = (Distribution*)(g.get_node(cauchy_dist));
  auto sampleNode = (oper::Sample*)(g.get_node(sample));

  // test gradient of log prob with respect to value
  {
    double actual_grad1 = 0;
    double actual_grad2 = 0;
    cauchy->gradient_log_prob_value(
        sampleNode->value, actual_grad1, actual_grad2);
    // Using https://www.wolframalpha.com/
    // D[log(PDF[CauchyDistribution[x0, s], x]), x]
    //                 = -2(x - x0) / (s^2 + (x - x0)^2)
    EXPECT_NEAR(
        actual_grad1,
        -2 * (x1 - X0) / (pow(SCALE, 2) + pow((x1 - X0), 2)),
        0.00001);
    // D[D[log(PDF[CauchyDistribution[x0, s], x]), x], x]
    //                 = (2 (-s^2 + (x - x0)^2))/(s^2 + (x - x0)^2)^2
    EXPECT_NEAR(
        actual_grad2,
        (2 * (-pow(SCALE, 2) + pow((x1 - X0), 2))) /
            pow((pow(SCALE, 2) + pow((x1 - X0), 2)), 2),
        0.00001);
  }

  // test gradient of log prob with respect to loc
  {
    double x0_grad1 = 4.321;
    double x0_grad2 = 0.1234;
    g.get_node(loc)->grad1 = x0_grad1;
    g.get_node(loc)->grad2 = x0_grad2;
    g.get_node(scale)->grad1 = 0;
    g.get_node(scale)->grad2 = 0;

    double actual_grad1 = 0;
    double actual_grad2 = 0;
    cauchy->gradient_log_prob_param(
        sampleNode->value, actual_grad1, actual_grad2);

    // D[log(PDF[CauchyDistribution[x0, s], x]), x0]
    //           = (2 (x - x0))/(s^2 + (x - x0)^2)
    double d1 = (2 * (x1 - X0)) / (pow(SCALE, 2) + pow((x1 - X0), 2));
    // D[D[log(PDF[CauchyDistribution[x0, s], x]), x0], x0]
    //           = (2 (-s^2 + (x - x0)^2))/(s^2 + (x - x0)^2)^2
    double d2 = (2 * (-pow(SCALE, 2) + pow((x1 - X0), 2))) /
        pow(pow(SCALE, 2) + pow((x1 - X0), 2), 2);

    // Use the first and second order chain rules
    double grad1 = d1 * x0_grad1;
    double grad2 = d2 * x0_grad1 * x0_grad1 + x0_grad2 * d1;

    EXPECT_NEAR(actual_grad1, grad1, 0.00001);
    EXPECT_NEAR(actual_grad2, grad2, 0.00001);
  }

  // test gradient of log prob with respect to scale
  {
    double scale_grad1 = 4.321;
    double scale_grad2 = 0.1234;
    g.get_node(loc)->grad1 = 0;
    g.get_node(loc)->grad2 = 0;
    g.get_node(scale)->grad1 = scale_grad1;
    g.get_node(scale)->grad2 = scale_grad2;

    double actual_grad1 = 0;
    double actual_grad2 = 0;
    cauchy->gradient_log_prob_param(
        sampleNode->value, actual_grad1, actual_grad2);

    // D[log(PDF[CauchyDistribution[x0, s], x]), s]
    //           = (-s^2 + (x - x0)^2)/(s (s^2 + (x - x0)^2))
    double d1 = (-pow(SCALE, 2) + pow((x1 - X0), 2)) /
        (SCALE * (pow(SCALE, 2) + pow((x1 - X0), 2)));
    // D[D[log(PDF[CauchyDistribution[x0, s], x]), s], s]
    //           = (s^4 - 4 s^2 (x - x0)^2 - (x - x0)^4) /
    //             (s^2 (s^2 + (x - x0)^2)^2)
    double d2 = (pow(SCALE, 4) - 4 * pow(SCALE, 2) * pow((x1 - X0), 2) -
                 pow((x1 - X0), 4)) /
        (pow(SCALE, 2) * pow(pow(SCALE, 2) + pow((x1 - X0), 2), 2));

    // Use the first and second order chain rules
    double grad1 = d1 * scale_grad1;
    double grad2 = d2 * scale_grad1 * scale_grad1 + scale_grad2 * d1;

    EXPECT_NEAR(actual_grad1, grad1, 0.00001);
    EXPECT_NEAR(actual_grad2, grad2, 0.00001);
  }

  // test backward gradient of logProb with respect to value
  {
    graph::DoubleMatrix actualGrad(0.0);
    double adjunct = 1.1;
    cauchy->backward_value(sampleNode->value, actualGrad, adjunct);

    double actual_grad1 = actualGrad.as_double();

    // Using https://www.wolframalpha.com/
    // D[log(PDF[CauchyDistribution[x0, s], x]), x]
    //                 = -2(x - x0) / (s^2 + (x - x0)^2)
    auto d1 = -2 * (x1 - X0) / (pow(SCALE, 2) + pow((x1 - X0), 2));
    EXPECT_NEAR(d1 * adjunct, actual_grad1, 0.00001);
  }

  // test backward gradient of logProb with respect to parameters
  {
    double adjunct = 1.1;
    g.get_node(loc)->back_grad1 = 0;
    g.get_node(scale)->back_grad1 = 0;
    cauchy->backward_param(sampleNode->value, adjunct);

    // D[log(PDF[CauchyDistribution[x0, s], x]), s]
    //           = (-s^2 + (x - x0)^2)/(s (s^2 + (x - x0)^2))
    double d1Scale = (-pow(SCALE, 2) + pow((x1 - X0), 2)) /
        (SCALE * (pow(SCALE, 2) + pow((x1 - X0), 2)));

    // D[log(PDF[CauchyDistribution[x0, s], x]), x0]
    //           = (2 (x - x0))/(s^2 + (x - x0)^2)
    double d1Loc = (2 * (x1 - X0)) / (pow(SCALE, 2) + pow((x1 - X0), 2));

    EXPECT_NEAR(d1Loc * adjunct, g.get_node(loc)->back_grad1, 0.00001);
    EXPECT_NEAR(d1Scale * adjunct, g.get_node(scale)->back_grad1, 0.00001);
  }
}
