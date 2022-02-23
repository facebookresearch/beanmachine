/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <stdexcept>
#include <tuple>

#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"

using namespace beanmachine;

TEST(testgraph, infer_arithmetic) {
  graph::Graph g;
  uint c1 = g.add_constant_probability(0.1);
  uint d1 = g.add_distribution(
      graph::DistributionType::BERNOULLI,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({c1}));
  uint o1 =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d1}));
  uint o2 =
      g.add_operator(graph::OperatorType::TO_POS_REAL, std::vector<uint>({o1}));
  uint c2 = g.add_constant_pos_real(0.8);
  uint o3 = g.add_operator(
      graph::OperatorType::MULTIPLY, std::vector<uint>({c2, o2}));
  uint c3 = g.add_constant_pos_real(0.1);
  uint o4 =
      g.add_operator(graph::OperatorType::ADD, std::vector<uint>({c3, o3}));
  uint d2 = g.add_distribution(
      graph::DistributionType::BERNOULLI_NOISY_OR,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({o4}));
  uint o5 =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d2}));
  // P(o5|o1=T) = 1 - exp(-.9)=0.5934 and P(o5|o1=F) = 1-exp(-.1)=0.09516
  // Since P(o1=T)=0.1 and P(o1=F)=0.9. Therefore P(o5=T,o1=T) = 0.05934,
  // P(o5=T,o1=F) = 0.08564 and P(o1=T | o5=T) = 0.4093
  g.observe(o5, true);
  g.query(o1);
  std::vector<std::vector<graph::NodeValue>> samples =
      g.infer(100, graph::InferenceType::GIBBS);
  int sum = 0;
  for (const auto& sample : samples) {
    const auto& s = sample.front();
    ASSERT_EQ(s.type, graph::AtomicType::BOOLEAN);
    sum += s._bool ? 1 : 0;
  }
  // less than 1 in million odds of failing these tests with correct infer
  EXPECT_LT(sum, 65);
  EXPECT_GT(sum, 15);
  // infer_mean should give almost exactly the same answer
  std::vector<double> means = g.infer_mean(100, graph::InferenceType::GIBBS);
  EXPECT_TRUE(std::abs(sum - int(means[0] * 100)) <= 1);
  // repeat the test with rejection sampling
  std::vector<std::vector<graph::NodeValue>> samples2 =
      g.infer(100, graph::InferenceType::REJECTION);
  sum = 0;
  for (const auto& sample : samples2) {
    const auto& s = sample.front();
    ASSERT_EQ(s.type, graph::AtomicType::BOOLEAN);
    sum += s._bool ? 1 : 0;
  }
  EXPECT_LT(sum, 65);
  EXPECT_GT(sum, 15);
  // infer_mean should give the same answer
  std::vector<double> means2 =
      g.infer_mean(100, graph::InferenceType::REJECTION);
  EXPECT_TRUE(std::abs(sum - int(means2[0] * 100)) <= 1);
}

TEST(testgraph, infer_bn) {
  graph::Graph g;
  // classic sprinkler BN, see the diagram here:
  // https://upload.wikimedia.org/wikipedia/commons/0/0e/SimpleBayesNet.svg
  torch::Tensor matrix1(2, 1);
  matrix1 << 0.8, 0.2;
  uint c1 = g.add_constant_col_simplex_matrix(matrix1);
  uint d1 = g.add_distribution(
      graph::DistributionType::TABULAR,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({c1}));
  uint RAIN =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d1}));
  torch::Tensor matrix2(2, 2);
  matrix2 << 0.6, 0.99, 0.4, 0.01;
  uint c2 = g.add_constant_col_simplex_matrix(matrix2);
  uint d2 = g.add_distribution(
      graph::DistributionType::TABULAR,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({c2, RAIN}));
  uint SPRINKLER =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d2}));
  torch::Tensor matrix3(2, 4);
  matrix3 << 1.0, 0.2, 0.1, 0.01, 0.0, 0.8, 0.9, 0.99;
  uint c3 = g.add_constant_col_simplex_matrix(matrix3);
  uint d3 = g.add_distribution(
      graph::DistributionType::TABULAR,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({c3, SPRINKLER, RAIN}));
  uint GRASSWET =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d3}));
  g.observe(GRASSWET, true);
  g.query(RAIN);
  uint n_iter = 100;
  const std::vector<std::vector<graph::NodeValue>>& samples =
      g.infer(n_iter, graph::InferenceType::REJECTION);
  ASSERT_EQ(samples.size(), n_iter);
  uint sum = 0;
  for (const auto& sample : samples) {
    const auto& s = sample.front();
    sum += s._bool ? 1 : 0;
  }
  // true probability is approx .3577
  // less than 1 in million odds of failing these tests with correct infer
  EXPECT_LT(sum, 60);
  EXPECT_GT(sum, 10);
  // using multiple chains
  const auto& all_samples =
      g.infer(n_iter, graph::InferenceType::REJECTION, 123, 2);
  ASSERT_EQ(all_samples.size(), 2);
  ASSERT_EQ(all_samples[0].size(), n_iter);
  ASSERT_EQ(all_samples[1].size(), n_iter);

  uint eqsum = 0;
  for (int i = 0; i < n_iter; i++) {
    const auto& s0 = all_samples[0][i].front();
    const auto& s1 = all_samples[1][i].front();
    // @lint-ignore CLANGTIDY
    eqsum += (s0._value == s1._bool) ? 1 : 0;
  }
  ASSERT_LT(eqsum, n_iter);

  const auto& all_means =
      g.infer_mean(n_iter, graph::InferenceType::REJECTION, 123, 2);
  ASSERT_EQ(all_means.size(), 2);
  ASSERT_NE(all_means[0].front(), all_means[1].front());
  ASSERT_GT(all_means[0].front(), 0.1);
  ASSERT_LT(all_means[0].front(), 0.6);
  ASSERT_GT(all_means[1].front(), 0.1);
  ASSERT_LT(all_means[1].front(), 0.6);
}

TEST(testgraph, clone_graph) {
  // This graph is not a meaningful model. It is designed to include all
  // types of nodes to test the copy constructor.
  graph::Graph g;
  // constants
  uint c_value = g.add_constant(true);
  uint c_real = g.add_constant(-2.5);
  uint c_natural_1 = g.add_constant((graph::natural_t)1);
  uint c_natural_2 = g.add_constant((graph::natural_t)2);
  uint c_prob = g.add_constant_probability(0.5);
  uint c_pos = g.add_constant_pos_real(2.5);
  uint c_neg = g.add_constant_neg_real(-1.5);

  torch::Tensor m0 = torch::Tensor::Constant(2, 1, 0.6);
  g.add_constant_probability_matrix(m0);
  torch::Tensor m1 = torch::Tensor::Identity(2, 2);
  g.add_constant_pos_matrix(m1);
  torch::Tensor m2 = torch::Tensor::Random(2, 2);
  g.add_constant_real_matrix(m2);
  torch::Tensor m3(2, 1);
  m3 << 0.2, 0.8;
  g.add_constant_col_simplex_matrix(m3);
  torch::Tensor m4(1, 2);
  m4 << true, false;
  g.add_constant_bool_matrix(m4);
  torch::Tensor m5(2, 1);
  m5 << 1, 2;
  g.add_constant_natural_matrix(m5);
  // distributions
  uint d_bernoulli = g.add_distribution(
      graph::DistributionType::BERNOULLI,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>{c_prob});
  uint d_bernoulli_or = g.add_distribution(
      graph::DistributionType::BERNOULLI_NOISY_OR,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>{c_pos});
  uint d_bernoulli_logit = g.add_distribution(
      graph::DistributionType::BERNOULLI_LOGIT,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>{c_real});
  uint d_beta = g.add_distribution(
      graph::DistributionType::BETA,
      graph::AtomicType::PROBABILITY,
      std::vector<uint>{c_pos, c_pos});
  uint d_binomial = g.add_distribution(
      graph::DistributionType::BINOMIAL,
      graph::AtomicType::NATURAL,
      std::vector<uint>{c_natural_2, c_prob});
  uint d_flat = g.add_distribution(
      graph::DistributionType::FLAT,
      graph::AtomicType::POS_REAL,
      std::vector<uint>{});
  uint d_normal = g.add_distribution(
      graph::DistributionType::NORMAL,
      graph::AtomicType::REAL,
      std::vector<uint>{c_real, c_pos});
  uint d_halfcauchy = g.add_distribution(
      graph::DistributionType::HALF_CAUCHY,
      graph::AtomicType::POS_REAL,
      std::vector<uint>{c_pos});
  uint d_studentt = g.add_distribution(
      graph::DistributionType::STUDENT_T,
      graph::AtomicType::REAL,
      std::vector<uint>{c_pos, c_real, c_pos});
  uint d_gamma = g.add_distribution(
      graph::DistributionType::GAMMA,
      graph::AtomicType::POS_REAL,
      std::vector<uint>{c_pos, c_pos});
  // operators
  uint o_sample_value = g.add_operator(
      graph::OperatorType::SAMPLE, std::vector<uint>{d_bernoulli});
  uint o_sample_real =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>{d_normal});
  uint o_sample_natural = g.add_operator(
      graph::OperatorType::SAMPLE, std::vector<uint>{d_binomial});
  uint o_sample_prob =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>{d_beta});
  uint o_sample_pos =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>{d_gamma});

  uint o_iidsample_value = g.add_operator(
      graph::OperatorType::IID_SAMPLE,
      std::vector<uint>{d_bernoulli, c_natural_1, c_natural_2});
  uint o_iidsample_real = g.add_operator(
      graph::OperatorType::IID_SAMPLE,
      std::vector<uint>{d_normal, c_natural_2, c_natural_2});
  uint o_iidsample_natural = g.add_operator(
      graph::OperatorType::IID_SAMPLE,
      std::vector<uint>{d_binomial, c_natural_2});
  uint o_iidsample_prob = g.add_operator(
      graph::OperatorType::IID_SAMPLE,
      std::vector<uint>{d_beta, c_natural_2, c_natural_1});
  uint o_iidsample_pos = g.add_operator(
      graph::OperatorType::IID_SAMPLE,
      std::vector<uint>{d_gamma, c_natural_2, c_natural_2});

  uint o_to_real = g.add_operator(
      graph::OperatorType::TO_REAL, std::vector<uint>{o_sample_pos});
  uint o_to_pos = g.add_operator(
      graph::OperatorType::TO_POS_REAL, std::vector<uint>{o_sample_prob});
  uint o_complement = g.add_operator(
      graph::OperatorType::COMPLEMENT, std::vector<uint>{o_sample_prob});
  uint o_negate =
      g.add_operator(graph::OperatorType::NEGATE, std::vector<uint>{c_real});
  uint o_exp =
      g.add_operator(graph::OperatorType::EXP, std::vector<uint>{c_real});
  uint o_expm1 = g.add_operator(
      graph::OperatorType::EXPM1, std::vector<uint>{o_sample_pos});
  uint o_log =
      g.add_operator(graph::OperatorType::LOG, std::vector<uint>{o_to_pos});
  uint o_log1pexp =
      g.add_operator(graph::OperatorType::LOG1PEXP, std::vector<uint>{o_log});
  uint o_log1mexp =
      g.add_operator(graph::OperatorType::LOG1MEXP, std::vector<uint>{c_neg});
  uint o_logsumexp = g.add_operator(
      graph::OperatorType::LOGSUMEXP, std::vector<uint>{c_real, o_sample_real});
  uint o_multiply = g.add_operator(
      graph::OperatorType::MULTIPLY,
      std::vector<uint>{c_real, o_sample_real, o_logsumexp});
  uint o_add = g.add_operator(
      graph::OperatorType::ADD,
      std::vector<uint>{c_real, o_sample_real, o_to_real});
  uint o_phi = g.add_operator(
      graph::OperatorType::PHI, std::vector<uint>{o_sample_real});
  uint o_logistic =
      g.add_operator(graph::OperatorType::LOGISTIC, std::vector<uint>{c_real});
  uint o_ifelse = g.add_operator(
      graph::OperatorType::IF_THEN_ELSE,
      std::vector<uint>{o_sample_bool, o_sample_pos, o_log1pexp});
  // factors
  uint f_expprod = g.add_factor(
      graph::FactorType::EXP_PRODUCT,
      std::vector<uint>{o_sample_real, o_sample_pos, o_to_pos});
  // observe and query
  g.observe(o_sample_real, 1.5);
  g.observe(o_sample_prob, 0.1);

  g.observe(o_iidsample_prob, m0);
  g.observe(o_iidsample_pos, m1);
  g.observe(o_iidsample_real, m2);
  g.observe(o_iidsample_bool, m4);
  g.observe(o_iidsample_natural, m5);

  g.query(o_multiply);
  g.query(o_add);
  g.query(o_ifelse);
  g.query(o_log1mexp);
  // copy and test
  graph::Graph g_copy(g);
  ASSERT_EQ(g.to_string(), g_copy.to_string());
}

TEST(testgraph, full_log_prob) {
  graph::Graph g;
  uint a = g.add_constant_pos_real(5.0);
  uint b = g.add_constant_pos_real(3.0);
  uint n = g.add_constant((graph::natural_t)10);
  uint beta = g.add_distribution(
      graph::DistributionType::BETA,
      graph::AtomicType::PROBABILITY,
      std::vector<uint>({a, b}));
  uint prob =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({beta}));
  uint binomial = g.add_distribution(
      graph::DistributionType::BINOMIAL,
      graph::AtomicType::NATURAL,
      std::vector<uint>({n, prob}));
  uint k = g.add_operator(
      graph::OperatorType::SAMPLE, std::vector<uint>({binomial}));
  g.observe(k, (graph::natural_t)8);
  g.query(prob);
  // Note: the posterior is p ~ Beta(13, 5).
  int num_samples = 10;
  auto conf = graph::InferConfig(true);
  auto samples = g.infer(num_samples, graph::InferenceType::NMC, 123, 2, conf);
  auto log_probs = g.get_log_prob();
  EXPECT_EQ(log_probs.size(), 2);
  EXPECT_EQ(log_probs[0].size(), num_samples);
  for (uint c = 0; c < 2; ++c) {
    for (uint i = 0; i < num_samples; ++i) {
      auto& s = samples[c][i][0];
      double l = log_probs[c][i];
      g.remove_observations();
      g.observe(k, (graph::natural_t)8);
      g.observe(prob, s._value);
      EXPECT_NEAR(g.full_log_prob(), l, 1e-3);
    }
  }
  g.remove_observations();
  g.observe(k, (graph::natural_t)8);
  g.observe(prob, 0.6);
  EXPECT_NEAR(g.full_log_prob(), -1.3344, 1e-3);
}

TEST(testgraph, bad_observations) {
  // Tests which demonstrate that we give errors for bad observations.
  torch::Tensor bool_matrix(1, 2);
  bool_matrix << true, false;
  torch::Tensor nat_matrix(1, 2);
  nat_matrix << 2, 3;
  torch::Tensor real_matrix(1, 2);
  real_matrix << 1.5, 2.5;
  graph::natural_t nat = 2;

  graph::Graph g;

  // Observe a bool to be a double, natural, matrix:
  uint c_prob = g.add_constant_probability(0.5);
  uint d_bernoulli = g.add_distribution(
      graph::DistributionType::BERNOULLI,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>{c_prob});
  uint o_sample_value = g.add_operator(
      graph::OperatorType::SAMPLE, std::vector<uint>{d_bernoulli});
  EXPECT_THROW(g.observe(o_sample_bool, 0.1), std::invalid_argument);
  EXPECT_THROW(g.observe(o_sample_bool, nat), std::invalid_argument);
  EXPECT_THROW(g.observe(o_sample_bool, bool_matrix), std::invalid_argument);
  EXPECT_THROW(g.observe(o_sample_bool, nat_matrix), std::invalid_argument);
  EXPECT_THROW(g.observe(o_sample_bool, real_matrix), std::invalid_argument);

  // Observe a bool(2, 1) to be a bool, double, natural, and (1, 2) matrices
  uint c_natural_1 = g.add_constant((graph::natural_t)1);
  uint c_natural_2 = g.add_constant((graph::natural_t)2);
  uint o_iid_value = g.add_operator(
      graph::OperatorType::IID_SAMPLE,
      std::vector<uint>{d_bernoulli, c_natural_2, c_natural_1});
  EXPECT_THROW(g.observe(o_iid_bool, false), std::invalid_argument);
  EXPECT_THROW(g.observe(o_iid_bool, 0.1), std::invalid_argument);
  EXPECT_THROW(g.observe(o_iid_bool, nat), std::invalid_argument);
  EXPECT_THROW(g.observe(o_iid_bool, bool_matrix), std::invalid_argument);
  EXPECT_THROW(g.observe(o_iid_bool, nat_matrix), std::invalid_argument);
  EXPECT_THROW(g.observe(o_iid_bool, real_matrix), std::invalid_argument);

  // Observe a natural to be bool, real, matrix
  uint d_binomial = g.add_distribution(
      graph::DistributionType::BINOMIAL,
      graph::AtomicType::NATURAL,
      std::vector<uint>{c_natural_2, c_prob});
  uint o_sample_natural = g.add_operator(
      graph::OperatorType::SAMPLE, std::vector<uint>{d_binomial});
  EXPECT_THROW(g.observe(o_sample_natural, true), std::invalid_argument);
  EXPECT_THROW(g.observe(o_sample_natural, 0.5), std::invalid_argument);
  EXPECT_THROW(g.observe(o_sample_natural, bool_matrix), std::invalid_argument);
  EXPECT_THROW(g.observe(o_sample_natural, nat_matrix), std::invalid_argument);
  EXPECT_THROW(g.observe(o_sample_natural, real_matrix), std::invalid_argument);

  // Observe a natural(2, 1) to be a bool, double, natural, and (1, 2) matrices
  uint o_iid_nat = g.add_operator(
      graph::OperatorType::IID_SAMPLE,
      std::vector<uint>{d_binomial, c_natural_2, c_natural_1});
  EXPECT_THROW(g.observe(o_iid_nat, false), std::invalid_argument);
  EXPECT_THROW(g.observe(o_iid_nat, 0.1), std::invalid_argument);
  EXPECT_THROW(g.observe(o_iid_nat, nat), std::invalid_argument);
  EXPECT_THROW(g.observe(o_iid_nat, bool_matrix), std::invalid_argument);
  EXPECT_THROW(g.observe(o_iid_nat, nat_matrix), std::invalid_argument);
  EXPECT_THROW(g.observe(o_iid_nat, real_matrix), std::invalid_argument);

  // Observe a real to be bool, natural, matrix:
  uint c_real = g.add_constant(-2.5);
  uint c_pos = g.add_constant_pos_real(2.5);
  uint d_normal = g.add_distribution(
      graph::DistributionType::NORMAL,
      graph::AtomicType::REAL,
      std::vector<uint>{c_real, c_pos});
  uint o_sample_real =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>{d_normal});
  EXPECT_THROW(g.observe(o_sample_real, true), std::invalid_argument);
  EXPECT_THROW(g.observe(o_sample_real, nat), std::invalid_argument);
  EXPECT_THROW(g.observe(o_sample_real, bool_matrix), std::invalid_argument);
  EXPECT_THROW(g.observe(o_sample_real, nat_matrix), std::invalid_argument);
  EXPECT_THROW(g.observe(o_sample_real, real_matrix), std::invalid_argument);

  // Observe a real(2, 1) to be a bool, double, natural, and (1, 2) matrices
  uint o_iid_real = g.add_operator(
      graph::OperatorType::IID_SAMPLE,
      std::vector<uint>{d_normal, c_natural_2, c_natural_1});
  EXPECT_THROW(g.observe(o_iid_real, false), std::invalid_argument);
  EXPECT_THROW(g.observe(o_iid_real, 0.1), std::invalid_argument);
  EXPECT_THROW(g.observe(o_iid_real, nat), std::invalid_argument);
  EXPECT_THROW(g.observe(o_iid_real, bool_matrix), std::invalid_argument);
  EXPECT_THROW(g.observe(o_iid_real, nat_matrix), std::invalid_argument);
  EXPECT_THROW(g.observe(o_iid_real, real_matrix), std::invalid_argument);
}

TEST(testgraph, infer_runtime_error) {
  graph::Graph g;
  auto two = g.add_constant((graph::natural_t)2);
  torch::Tensor real_matrix(1, 1);
  real_matrix << 1.0;
  auto matrix = g.add_constant_real_matrix(real_matrix);
  // index out of bounds during runtime
  auto indexed_matrix =
      g.add_operator(graph::OperatorType::INDEX, {matrix, two});
  g.query(indexed_matrix);

  int num_samples = 10;
  int seed = 19;
  // test with one chain
  EXPECT_THROW(
      g.infer(num_samples, graph::InferenceType::NMC), std::runtime_error);
  // test with threads from multiple chains
  EXPECT_THROW(
      g.infer(num_samples, graph::InferenceType::NMC, seed, 2),
      std::runtime_error);
}
