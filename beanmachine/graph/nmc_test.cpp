// Copyright (c) Facebook, Inc. and its affiliates.
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "beanmachine/graph/graph.h"

using namespace beanmachine::graph;

TEST(testnmc, beta_binomial) {
  Graph g;
  uint a = g.add_constant_pos_real(5.0);
  uint b = g.add_constant_pos_real(3.0);
  uint n = g.add_constant((natural_t) 10);
  uint beta = g.add_distribution(
    DistributionType::BETA, AtomicType::PROBABILITY, std::vector<uint>({a, b}));
  uint prob = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({beta}));
  uint binomial = g.add_distribution(
    DistributionType::BINOMIAL, AtomicType::NATURAL, std::vector<uint>({n, prob}));
  uint k = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({binomial}));
  g.observe(k, (natural_t) 8);
  g.query(prob);
  // Note: the posterior is p ~ Beta(13, 5).
  int num_samples = 10000;
  std::vector<std::vector<AtomicValue>> samples =
      g.infer(num_samples, InferenceType::NMC);
  double sum = 0;
  double sumsq = 0;
  for (const auto& sample : samples) {
    const auto& s = sample.front();
    ASSERT_EQ(s.type, AtomicType::PROBABILITY);
    sum += s._double;
    sumsq += s._double * s._double;
  }
  double mean = sum / num_samples;
  double var = sumsq / num_samples - mean * mean;
  EXPECT_NEAR(mean, 13.0 / 18.0, 1e-2);
  EXPECT_NEAR(var, 13.0 * 5.0 / (18.0 * 18.0 * 19.0), 1e-3);
}

TEST(testnmc, net_norad) {
  // The NetNORAD model is as follows:
  // We have a number of components s.t. the i th one has a drop rate drop_i
  // We observe packets sent on multiple paths criss-crossing these components
  // For each path we observe the number sent and number dropped/received
  // We model the drop rate of a path as `1 - product_i (1 - drop_i)` for i in path
  // And we are trying to infer drop_i for each component
  // The prior on `drop_i ~ Beta(.0001, 100)` i.e. 1 in million odds
  // The prior on `pkts dropped on a path ~ Binomial(pkts sent, path-drop-rate)`
  Graph g;
  uint a = g.add_constant_pos_real(0.0001);
  uint b = g.add_constant_pos_real(100.0);
  uint drop_prior = g.add_distribution(
      DistributionType::BETA, AtomicType::PROBABILITY, std::vector<uint>{a, b});
  std::vector<uint> comp_rates; // complement of the drop rates
  for (int i=0; i<4; i++) {
    uint drop = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{drop_prior});
    g.query(drop);
    uint comp = g.add_operator(OperatorType::COMPLEMENT, std::vector<uint>{drop});
    comp_rates.push_back(comp);
  }
  // for each path the pkts sent, pkts recvd, component ids
  std::vector<std::tuple<uint, uint, std::vector<uint>>> paths = {
      {200, 200, {0, 1}},
      {200, 180, {1, 2}},
      {200, 170, {2, 3}},
      {200, 199, {0, 1, 3}}
  };
  for (const auto& path : paths) {
    uint pkts_sent = g.add_constant((natural_t) std::get<0>(path));
    std::vector<uint> path_comp_rates;
    for (uint id: std::get<2>(path)) {
      path_comp_rates.push_back(comp_rates[id]);
    }
    uint prod = g.add_operator(OperatorType::MULTIPLY, path_comp_rates);
    // path_drop_rate = 1 - product_{i | i in path} (1 - drop_i)
    uint path_drop_rate = g.add_operator(OperatorType::COMPLEMENT, std::vector<uint>{prod});
    uint path_dist = g.add_distribution(
        DistributionType::BINOMIAL,
        AtomicType::NATURAL,
        std::vector<uint>{pkts_sent, path_drop_rate});
    uint pkts_dropped = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{path_dist});
    g.observe(pkts_dropped, (natural_t)(std::get<0>(path) - std::get<1>(path)));
  }
  const std::vector<double>& means = g.infer_mean(10000, InferenceType::NMC);
  // component 0 is less than 1 because it has fewer dropped packets
  EXPECT_LT(means[0], means[1]);
  // 1 is less than 3 because it has fewer dropped packets
  EXPECT_LT(means[1], means[3]);
  // 2 is 100x of 1 because all paths involving 2 have high dropped packets
  EXPECT_LT(10 * means[1], means[2]);
}

TEST(testnmc, normal_normal) {
  Graph g;
  auto mean0 = g.add_constant(0.0);
  auto sigma0 = g.add_constant_pos_real(5.0);
  auto dist0 = g.add_distribution(DistributionType::NORMAL, AtomicType::REAL, std::vector<uint>{mean0, sigma0});
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist0});
  auto x_sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{x, x});
  auto sigma1 = g.add_constant_pos_real(10.0);
  auto dist1 = g.add_distribution(DistributionType::NORMAL, AtomicType::REAL, std::vector<uint>{x, sigma1});
  auto y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist1});
  g.observe(y, 100.0);
  g.query(x);
  g.query(x_sq);
  const std::vector<double>& means = g.infer_mean(10000, InferenceType::NMC);
  // posterior of x is N(20, sqrt(20))
  EXPECT_NEAR(means[0], 20, 0.1);
  EXPECT_NEAR(means[1] - means[0]*means[0], 20, 1.0);
}
