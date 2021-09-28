// Copyright 2004-present Facebook. All Rights Reserved.
#include <gtest/gtest.h>
#include <random>

#include "beanmachine/graph/global/global_mh.h"
#include "beanmachine/graph/graph.h"

using namespace beanmachine;
using namespace graph;

TEST(testglobal, rw_normal_normal) {
  /*
  p1 ~ Normal(0, 1)
  p2 ~ Normal(p1, 1)
  p2 observed as 0.5
  posterior is Normal(0.25, 0.5)
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
  RandomWalkMH mh = RandomWalkMH(g, 0.5);
  std::vector<std::vector<NodeValue>> samples = mh.infer(10000, seed);
  EXPECT_EQ(samples.size(), 10000);

  double mean = 0;
  for (int i = 0; i < samples.size(); i++) {
    mean += samples[i][0]._double;
  }
  mean /= samples.size();
  EXPECT_NEAR(mean, 0.25, 0.01);
}

TEST(testglobal, rw_distant_normal_normal) {
  /*
  p1 ~ Normal(100, 1)
  p2 ~ Normal(p1, 1)
  p2 observed as 0.5
  posterior is Normal(50.5, 0.5)
  */
  Graph g;
  uint hundred = g.add_constant(100.0);
  uint one = g.add_constant_pos_real(1.0);

  uint norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {hundred, one});
  uint sample = g.add_operator(OperatorType::SAMPLE, {norm_dist});

  uint norm_norm_dist = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, {sample, one});
  uint obs = g.add_operator(OperatorType::SAMPLE, {norm_norm_dist});

  g.observe(obs, 0.5);
  g.query(sample);

  uint seed = 17;
  RandomWalkMH mh = RandomWalkMH(g, 0.5);
  std::vector<std::vector<NodeValue>> samples = mh.infer(10000, seed);
  EXPECT_EQ(samples.size(), 10000);

  double mean = 0;
  for (int i = 0; i < samples.size(); i++) {
    mean += samples[i][0]._double;
  }
  mean /= samples.size();
  EXPECT_NEAR(mean, 50.5, 0.5);
}

TEST(testglobal, rw_gamma_gamm) {
  /*
  p1 ~ Gamma(2, 2)
  p2 ~ Gamma(1, p1)
  p2 observed as 2
  posterior is Gamma(3, 4)
  */
  Graph g;
  uint two = g.add_constant_pos_real(2.0);
  uint one = g.add_constant_pos_real(1.0);

  uint gamma_p_dist = g.add_distribution(
      DistributionType::GAMMA, AtomicType::POS_REAL, {two, two});
  uint gamma_p = g.add_operator(OperatorType::SAMPLE, {gamma_p_dist});

  uint gamma_dist = g.add_distribution(
      DistributionType::GAMMA, AtomicType::POS_REAL, {one, gamma_p});
  uint obs = g.add_operator(OperatorType::SAMPLE, {gamma_dist});

  g.observe(obs, 2.0);
  g.query(gamma_p);
  g.customize_transformation(TransformType::LOG, {gamma_p});

  uint seed = 17;
  RandomWalkMH mh = RandomWalkMH(g, 0.5);
  std::vector<std::vector<NodeValue>> samples = mh.infer(10000, seed);
  EXPECT_EQ(samples.size(), 10000);

  double mean = 0;
  for (int i = 0; i < samples.size(); i++) {
    mean += samples[i][0]._double;
  }
  mean /= samples.size();
  EXPECT_NEAR(mean, 0.75, 0.01);
}
