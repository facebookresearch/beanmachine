// Copyright 2004-present Facebook. All Rights Reserved.
#include <gtest/gtest.h>

#include "beanmachine/graph/global/nuts.h"
#include "beanmachine/graph/graph.h"

using namespace beanmachine;
using namespace graph;

TEST(testglobal, nuts_normal_normal) {
  /*
  p1 ~ Normal(0, 1)
  p2 ~ Normal(p1, 1)
  p2 observed as 0.5
  posterior is Normal(0.25, 0.5)
  expected mean is 0.25
  */
  Graph g;
  uint zero = g.add_constant(0.0);
  uint one = g.add_constant_pos_real(1.0);

  uint norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {zero, one});
  uint sample = g.add_operator(OperatorType::SAMPLE, {norm_dist});

  uint norm_norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {sample, one});
  uint obs = g.add_operator(OperatorType::SAMPLE, {norm_norm_dist});

  g.observe(obs, 0.5);
  g.query(sample);

  uint seed = 17;
  NUTS mh = NUTS(g);
  std::vector<std::vector<NodeValue>> samples = mh.infer(2000, seed, 1000);

  double mean = 0;
  for (int i = 0; i < samples.size(); i++) {
    mean += samples[i][0]._double;
  }
  mean /= samples.size();
  /*
  posterior is Normal(0.25, 0.5)
  expected mean is 0.25
  */
  EXPECT_NEAR(mean, 0.25, 0.02);
}

TEST(testglobal, nuts_gamma_gamma) {
  /*
  p1 ~ Gamma(2, 2)
  p2 ~ Gamma(1.5, p1)
  p2 observed as 2
  posterior is Gamma(3.5, 4)
  */
  Graph g;
  uint onepointfive = g.add_constant_pos_real(1.5);
  uint two = g.add_constant_pos_real(2.0);

  uint gamma_dist = g.add_distribution(
      DistributionType::GAMMA, AtomicType::POS_REAL, {two, two});
  uint sample = g.add_operator(OperatorType::SAMPLE, {gamma_dist});

  uint gamma_gamma_dist = g.add_distribution(
      DistributionType::GAMMA, AtomicType::POS_REAL, {onepointfive, sample});
  uint obs1 = g.add_operator(OperatorType::SAMPLE, {gamma_gamma_dist});
  g.customize_transformation(TransformType::LOG, {sample});

  g.observe(obs1, 2.0);
  g.query(sample);

  uint seed = 17;
  NUTS mh = NUTS(g);
  std::vector<std::vector<NodeValue>> samples = mh.infer(2000, seed, 1000);
  EXPECT_EQ(samples.size(), 2000);
  double mean = 0;
  for (int i = 0; i < samples.size(); i++) {
    mean += samples[i][0]._double;
  }
  mean /= samples.size();
  /*
  true posterior is Gamma(3.5, 4)
  mean should be 3.4 / 4 = 0.875
  */
  EXPECT_NEAR(mean, 0.875, 0.03);
}

TEST(testglobal, global_nuts_half_cauchy) {
  /*
  p1 ~ HalfCauchy(1)
  expected median of p1 is 1.0
  */
  Graph g;
  uint one = g.add_constant_pos_real(1.0);

  uint sigma_prior = g.add_distribution(
      DistributionType::HALF_CAUCHY, AtomicType::POS_REAL, {one});
  uint sigma_sample = g.add_operator(OperatorType::SAMPLE, {sigma_prior});
  g.customize_transformation(TransformType::LOG, {sigma_sample});
  g.query(sigma_sample);

  uint seed = 17;
  uint num_samples = 5000;
  NUTS mh = NUTS(g);
  std::vector<std::vector<NodeValue>> samples =
      mh.infer(num_samples, seed, num_samples);
  EXPECT_EQ(samples.size(), num_samples);

  double expected_median = 1.0;
  double above_expected_median = 0;
  for (int i = 0; i < samples.size(); i++) {
    if (samples[i][0]._double > expected_median) {
      above_expected_median++;
    }
  }
  // check that ~50% of the samples are greater than the expected median
  EXPECT_NEAR(above_expected_median / num_samples, 0.5, 0.02);
}

TEST(testglobal, global_nuts_mixed) {
  /*
  Test multiple queries and global inference
  p1 ~ Gamma(2, 2)
  p2 ~ Gamma(1.5, p1)
  p2 observed as 2
  posterior is Gamma(3.5, 4)
  expected mean is 3.5 / 4 = 0.875

  p3 ~ Normal(0, 1)
  p4 ~ Normal(p1, 1)
  p5 ~ Normal(p1, 1)
  p4 observed as 0.5
  p5 observed as 1.5
  posterior is Normal(0.66.., 0.5)
  expected mean is 0.66...
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
  g.customize_transformation(TransformType::LOG, {gamma_sample});

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
  NUTS mh = NUTS(g);
  std::vector<std::vector<NodeValue>> samples =
      mh.infer(2000, seed, 2000, false);
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

TEST(testglobal, global_nuts_halfnormal) {
  /*
  a ~ Normal(1, 1)
  b ~ HalfNormal(2)
  c ~ Normal(a + b, 1)
  posterior mean should be centered around 1.69
  as verified by Stan and Python BM
  */
  Graph g;
  uint one = g.add_constant(1.0);
  uint one_pos = g.add_constant_pos_real(1.0);
  uint two_pos = g.add_constant_pos_real(2.0);

  uint a_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {one, one_pos});
  uint a_sample = g.add_operator(OperatorType::SAMPLE, {a_dist});

  uint b_dist = g.add_distribution(
      DistributionType::HALF_NORMAL, AtomicType::POS_REAL, {two_pos});
  uint b_sample = g.add_operator(OperatorType::SAMPLE, {b_dist});
  uint b_sample_real = g.add_operator(OperatorType::TO_REAL, {b_sample});

  uint c_mean = g.add_operator(OperatorType::ADD, {a_sample, b_sample_real});
  uint c_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {c_mean, one_pos});
  uint c_sample = g.add_operator(OperatorType::SAMPLE, {c_dist});

  g.observe(c_sample, 5.0);
  g.query(a_sample);

  uint seed = 31;
  NUTS mh = NUTS(g);
  std::vector<std::vector<NodeValue>> samples = mh.infer(2000, seed, 2000);

  double mean = 0;
  for (int i = 0; i < samples.size(); i++) {
    mean += samples[i][0]._double;
  }
  mean /= samples.size();
  EXPECT_NEAR(mean, 1.69, 0.04);
}
