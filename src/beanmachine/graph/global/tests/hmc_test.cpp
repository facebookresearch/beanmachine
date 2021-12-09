/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "beanmachine/graph/global/hmc.h"
#include "beanmachine/graph/global/tests/conjugate_util_test.h"
#include "beanmachine/graph/graph.h"

using namespace beanmachine;
using namespace graph;

TEST(testglobal, global_hmc_normal_normal) {
  Graph g;
  build_normal_normal_model(g);
  HMC mh = HMC(g, 1.0, 1.0);
  test_normal_normal_model(mh);
}

TEST(testglobal, global_hmc_gamma_gamma) {
  Graph g;
  build_gamma_gamma_model(g);
  HMC mh = HMC(g, 1.0, 0.5);
  test_gamma_gamma_model(mh);
}

TEST(testglobal, global_hmc_half_cauchy) {
  /*
  p1 ~ HalfCauchy(1)
  expected median of p1 is 1.0
  */
  Graph g;
  uint one = g.add_constant_pos_real(1.0);

  uint sigma_prior = g.add_distribution(
      DistributionType::HALF_CAUCHY, AtomicType::POS_REAL, {one});
  uint sigma_sample = g.add_operator(OperatorType::SAMPLE, {sigma_prior});
  g.query(sigma_sample);

  uint seed = 17;
  uint num_samples = 10000;
  HMC mh = HMC(g, 1.0, 1.0);
  std::vector<std::vector<NodeValue>> samples = mh.infer(num_samples, seed);
  EXPECT_EQ(samples.size(), 10000);

  double expected_median = 1.0;
  double above_expected_median = 0;
  for (int i = 0; i < samples.size(); i++) {
    if (samples[i][0]._double > expected_median) {
      above_expected_median++;
    }
  }
  // check that ~50% of the samples are greater than the median
  EXPECT_NEAR(above_expected_median / num_samples, 0.5, 0.02);
}

TEST(testglobal, global_hmc_warmup_normal_normal) {
  /*
  p1 ~ Normal(0, 1)
  p2 ~ Normal(p1, 1)
  p3 ~ Normal(p1, 1)
  p2 observed as 0.5
  p3 observed as 1.5
  posterior is Normal(0.66.., 0.5)
  */
  Graph g;
  uint zero = g.add_constant(0.0);
  uint one = g.add_constant_pos_real(1.0);

  uint norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {zero, one});
  uint sample = g.add_operator(OperatorType::SAMPLE, {norm_dist});

  uint norm_norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {sample, one});
  uint obs1 = g.add_operator(OperatorType::SAMPLE, {norm_norm_dist});
  uint obs2 = g.add_operator(OperatorType::SAMPLE, {norm_norm_dist});

  g.observe(obs1, 0.5);
  g.observe(obs2, 1.5);
  g.query(sample);

  uint seed = 17;
  HMC mh = HMC(g, 1.0, 1.0);
  std::vector<std::vector<NodeValue>> samples = mh.infer(500, seed, 500);
  EXPECT_EQ(samples.size(), 500);

  double mean = 0;
  for (int i = 0; i < samples.size(); i++) {
    mean += samples[i][0]._double;
  }
  mean /= samples.size();
  EXPECT_NEAR(mean, 2.0 / 3, 0.02);
}

TEST(testglobal, global_hmc_warmup) {
  /*
  Test multiple queries and global inference
  p1 ~ Gamma(2, 2)
  p2 ~ Gamma(1.5, p1)
  p2 observed as 2
  posterior is Gamma(3.5, 4)

  p3 ~ Normal(0, 1)
  p4 ~ Normal(p1, 1)
  p5 ~ Normal(p1, 1)
  p4 observed as 0.5
  p5 observed as 1.5
  posterior is Normal(0.66.., 0.5)
  */
  Graph g;
  uint onepointfive = g.add_constant_pos_real(1.5);
  uint two = g.add_constant_pos_real(2.0);

  uint gamma_dist = g.add_distribution(
      DistributionType::GAMMA, AtomicType::POS_REAL, {two, two});
  uint gamma_sample = g.add_operator(OperatorType::SAMPLE, {gamma_dist});

  uint gamma_gamma_dist = g.add_distribution(
      DistributionType::GAMMA,
      AtomicType::POS_REAL,
      {onepointfive, gamma_sample});
  uint gamma_obs1 = g.add_operator(OperatorType::SAMPLE, {gamma_gamma_dist});

  g.observe(gamma_obs1, 2.0);
  g.query(gamma_sample);

  uint zero = g.add_constant(0.0);
  uint one = g.add_constant_pos_real(1.0);

  uint norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {zero, one});
  uint sample = g.add_operator(OperatorType::SAMPLE, {norm_dist});

  uint norm_norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {sample, one});
  uint obs1 = g.add_operator(OperatorType::SAMPLE, {norm_norm_dist});
  uint obs2 = g.add_operator(OperatorType::SAMPLE, {norm_norm_dist});

  g.observe(obs1, 0.5);
  g.observe(obs2, 1.5);
  g.query(sample);

  uint seed = 17;
  HMC mh = HMC(g, 1.0, 0.1);
  std::vector<std::vector<NodeValue>> samples =
      mh.infer(2000, seed, 1000, false);
  EXPECT_EQ(samples.size(), 2000);

  double mean = 0;
  for (int i = 0; i < samples.size(); i++) {
    mean += samples[i][1]._double;
  }
  mean /= samples.size();
  EXPECT_NEAR(mean, 2.0 / 3, 0.02);

  mean = 0;
  for (int i = 0; i < samples.size(); i++) {
    mean += samples[i][0]._double;
  }
  mean /= samples.size();
  EXPECT_NEAR(mean, 0.875, 0.03);
}
