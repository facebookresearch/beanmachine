// Copyright (c) Facebook, Inc. and its affiliates.
#include <gtest/gtest.h>
#include <random>

#include "beanmachine/graph/graph.h"

using namespace beanmachine::graph;

TEST(testnmc, beta_binomial) {
  Graph g;
  uint a = g.add_constant_pos_real(5.0);
  uint b = g.add_constant_pos_real(3.0);
  uint n = g.add_constant((natural_t)10);
  uint beta = g.add_distribution(
      DistributionType::BETA,
      AtomicType::PROBABILITY,
      std::vector<uint>({a, b}));
  uint prob = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({beta}));
  uint binomial = g.add_distribution(
      DistributionType::BINOMIAL,
      AtomicType::NATURAL,
      std::vector<uint>({n, prob}));
  uint k = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({binomial}));
  g.observe(k, (natural_t)8);
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
  // We model the drop rate of a path as `1 - product_i (1 - drop_i)` for i in
  // path And we are trying to infer drop_i for each component The prior on
  // `drop_i ~ Beta(.0001, 100)` i.e. 1 in million odds The prior on `pkts
  // dropped on a path ~ Binomial(pkts sent, path-drop-rate)`
  Graph g;
  uint a = g.add_constant_pos_real(0.0001);
  uint b = g.add_constant_pos_real(100.0);
  uint drop_prior = g.add_distribution(
      DistributionType::BETA, AtomicType::PROBABILITY, std::vector<uint>{a, b});
  std::vector<uint> comp_rates; // complement of the drop rates
  for (int i = 0; i < 4; i++) {
    uint drop =
        g.add_operator(OperatorType::SAMPLE, std::vector<uint>{drop_prior});
    g.query(drop);
    uint comp =
        g.add_operator(OperatorType::COMPLEMENT, std::vector<uint>{drop});
    comp_rates.push_back(comp);
  }
  // for each path the pkts sent, pkts recvd, component ids
  std::vector<std::tuple<uint, uint, std::vector<uint>>> paths = {
      {200, 200, {0, 1}},
      {200, 180, {1, 2}},
      {200, 170, {2, 3}},
      {200, 199, {0, 1, 3}}};
  for (const auto& path : paths) {
    uint pkts_sent = g.add_constant((natural_t)std::get<0>(path));
    std::vector<uint> path_comp_rates;
    for (uint id : std::get<2>(path)) {
      path_comp_rates.push_back(comp_rates[id]);
    }
    uint prod = g.add_operator(OperatorType::MULTIPLY, path_comp_rates);
    // path_drop_rate = 1 - product_{i | i in path} (1 - drop_i)
    uint path_drop_rate =
        g.add_operator(OperatorType::COMPLEMENT, std::vector<uint>{prod});
    uint path_dist = g.add_distribution(
        DistributionType::BINOMIAL,
        AtomicType::NATURAL,
        std::vector<uint>{pkts_sent, path_drop_rate});
    uint pkts_dropped =
        g.add_operator(OperatorType::SAMPLE, std::vector<uint>{path_dist});
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
  auto dist0 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{mean0, sigma0});
  auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist0});
  auto x_sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{x, x});
  auto sigma1 = g.add_constant_pos_real(10.0);
  auto dist1 = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, std::vector<uint>{x, sigma1});
  auto y = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist1});
  g.observe(y, 100.0);
  g.query(x);
  g.query(x_sq);
  const std::vector<double>& means = g.infer_mean(10000, InferenceType::NMC);
  // posterior of x is N(20, sqrt(20))
  EXPECT_NEAR(means[0], 20, 0.1);
  EXPECT_NEAR(means[1] - means[0] * means[0], 20, 1.0);
}

TEST(testnmc, flat_normal) {
  // This test learns both the mean and standard deviation
  Graph g;
  auto mean_prior = g.add_distribution(
      DistributionType::FLAT, AtomicType::REAL, std::vector<uint>{});
  auto mean =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{mean_prior});
  auto sigma_prior = g.add_distribution(
      DistributionType::FLAT, AtomicType::POS_REAL, std::vector<uint>{});
  auto sigma =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{sigma_prior});
  auto likelihood = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{mean, sigma});
  // we will sample some values from the normal distribution and try to infer
  // the empirical mean and std
  int num_obs = 20;
  std::normal_distribution<double> dist(4.3, 5.1);
  std::mt19937 gen(314521);
  double emp_mean = 0;
  double emp_var = 0;
  for (int i = 0; i < num_obs; i++) {
    double val = dist(gen);
    auto n =
        g.add_operator(OperatorType::SAMPLE, std::vector<uint>{likelihood});
    g.observe(n, val);
    emp_mean += val / num_obs;
    emp_var += val * val / (num_obs - 1);
  }
  emp_var -= emp_mean * emp_mean * num_obs /
      (num_obs - 1); // unbiased estimator of sample variance
  g.query(mean);
  g.query(sigma);
  const std::vector<double>& post_means =
      g.infer_mean(10000, InferenceType::NMC);
  EXPECT_NEAR(post_means[0], emp_mean, 0.1);
  EXPECT_NEAR(post_means[1], std::sqrt(emp_var), 1.0);
}

TEST(testnmc, gmm_two_components) {
  // Test two component Gaussian Mixture Model
  // tau ~ Beta(1, 1)
  // mu_{false|true} ~ Normal(0, 100)
  // c_i ~ Bernoulli(tau)
  // X_i ~ Normal(mu_{c_i}, 1) are observed with values [1.0, 8.0, 11.0]
  // and observe c_0 = false (to break symmetry)
  // posterior of mu_false will be close to 1 and
  // posterior of mu_true will be close to 9.5
  // posterior of tau is Beta (3, 2) which has a mean of 0.6
  Graph g;
  auto zero = g.add_constant(0.0);
  auto one = g.add_constant_pos_real(1.0);
  auto hundred = g.add_constant_pos_real(100.0);
  auto tau_prior = g.add_distribution(
      DistributionType::BETA,
      AtomicType::PROBABILITY,
      std::vector<uint>{one, one});
  auto tau = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{tau_prior});
  auto c_prior = g.add_distribution(
      DistributionType::BERNOULLI, AtomicType::BOOLEAN, std::vector<uint>{tau});
  auto mu_prior = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{zero, hundred});
  auto mu_false =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{mu_prior});
  auto mu_true =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{mu_prior});
  const std::vector<double> X_DATA = {1.0, 8.0, 11.0};
  for (const auto x_val : X_DATA) {
    auto c_i = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{c_prior});
    if (x_val == 1.0) {
      g.observe(c_i, false);
    }
    auto mu = g.add_operator(
        OperatorType::IF_THEN_ELSE, std::vector<uint>{c_i, mu_true, mu_false});
    auto x_prior = g.add_distribution(
        DistributionType::NORMAL, AtomicType::REAL, std::vector<uint>{mu, one});
    auto x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{x_prior});
    g.observe(x, x_val);
  }
  g.query(tau);
  g.query(mu_false);
  g.query(mu_true);
  const std::vector<double>& post_means =
      g.infer_mean(1000, InferenceType::NMC);
  EXPECT_NEAR(post_means[0], 0.6, 0.01);
  EXPECT_NEAR(post_means[1], 1.0, 0.1);
  EXPECT_NEAR(post_means[2], 9.5, 0.1);
}

TEST(testnmc, infinite_grad) {
  /*
  This is a manual construction of a modified version of the BMA++ model.
  In practice, this model structure is not used, but it could produce
  float-divide-by-zero error. Because exp() overflows to Inf, making the
  grad1 and grad2 both infinite, then an invalid Normal proposer with
  sigma = 0.
  This test case is used to test if the bug is fixed.
  The model structure is:
      with yi and sei observed for each observation i
      - likelihood
          yi ~ Normal(yhat_i, sei)
          yhat_i = exp(fere_i) if sign_i else -exp(fere_i)
          sign_i ~ Bernoulli(prob_sign)
          fere_i = fixed_effect + random_effect(i)
          random_effect ~ Normal(0, re_scale)
      - priors
          fixed_effect ~ Normal(0, 2)
          re_scale ~ Half-Cauchy(1)
          prob_sign ~ Beta(1, 1)
  */
  Graph g;
  uint zero = g.add_constant(0.0);
  uint one = g.add_constant_pos_real(1.0);
  uint two = g.add_constant_pos_real(2.0);
  uint sei = g.add_constant_pos_real(0.15);
  std::vector<double> y = {
      0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5};
  std::vector<uint> re_id = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};

  uint beta_prior = g.add_distribution(
      DistributionType::BETA,
      AtomicType::PROBABILITY,
      std::vector<uint>({one, one}));
  uint normal_prior = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>({zero, two}));
  uint halfcauchy_prior = g.add_distribution(
      DistributionType::HALF_CAUCHY,
      AtomicType::POS_REAL,
      std::vector<uint>({one}));

  uint prob_sign =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>({beta_prior}));
  uint fe =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>({normal_prior}));
  uint re_scale = g.add_operator(
      OperatorType::SAMPLE, std::vector<uint>({halfcauchy_prior}));

  uint sign_dist = g.add_distribution(
      DistributionType::BERNOULLI,
      AtomicType::BOOLEAN,
      std::vector<uint>({prob_sign}));
  uint re_dist = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>({zero, re_scale}));
  uint re0_value =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>({re_dist}));
  uint re1_value =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>({re_dist}));
  std::vector<uint> re_values = {re0_value, re1_value};

  for (uint i = 0; i < y.size(); i++) {
    uint re = re_values[re_id[i]];
    uint fere = g.add_operator(OperatorType::ADD, std::vector<uint>({fe, re}));
    uint sign =
        g.add_operator(OperatorType::SAMPLE, std::vector<uint>({sign_dist}));
    uint yhat_pos =
        g.add_operator(OperatorType::EXP, std::vector<uint>({fere}));
    uint yhat_pos_real =
        g.add_operator(OperatorType::TO_REAL, std::vector<uint>({yhat_pos}));
    uint yhat_neg =
        g.add_operator(OperatorType::NEGATE, std::vector<uint>({yhat_pos_real}));
    uint yhat = g.add_operator(
        OperatorType::IF_THEN_ELSE,
        std::vector<uint>({sign, yhat_pos_real, yhat_neg}));
    uint y_dist = g.add_distribution(
        DistributionType::NORMAL,
        AtomicType::REAL,
        std::vector<uint>({yhat, sei}));
    uint yi = g.add_operator(OperatorType::SAMPLE, std::vector<uint>({y_dist}));
    g.observe(yi, (double)y[i]);
  }
  g.query(prob_sign);
  g.query(fe);
  g.query(re_scale);

  int num_samples = 1000;
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
  EXPECT_NEAR(mean, 0.5, 0.1);
  EXPECT_LT(0.0, var);
}
