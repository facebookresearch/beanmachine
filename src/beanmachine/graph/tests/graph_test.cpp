// Copyright (c) Facebook, Inc. and its affiliates.
#include <array>
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
  std::vector<std::vector<graph::AtomicValue>> samples =
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
  std::vector<std::vector<graph::AtomicValue>> samples2 =
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
  Eigen::MatrixXd matrix1(2, 1);
  matrix1 << 0.8,
             0.2;
  uint c1 = g.add_constant_col_simplex_matrix(matrix1);
  uint d1 = g.add_distribution(
      graph::DistributionType::TABULAR,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({c1}));
  uint RAIN =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d1}));
  Eigen::MatrixXd matrix2(2, 2);
  matrix2 << 0.6, 0.99,
             0.4, 0.01;
  uint c2 = g.add_constant_col_simplex_matrix(matrix2);
  uint d2 = g.add_distribution(
      graph::DistributionType::TABULAR,
      graph::AtomicType::BOOLEAN,
      std::vector<uint>({c2, RAIN}));
  uint SPRINKLER =
      g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>({d2}));
  Eigen::MatrixXd matrix3(2, 4);
  matrix3 << 1.0, 0.2, 0.1, 0.01,
             0.0, 0.8, 0.9, 0.99;
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
  const std::vector<std::vector<graph::AtomicValue>>& samples =
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
    // @lint-ignore HOWTOEVEN CLANGTIDY
    eqsum += (s0._bool == s1._bool) ? 1 : 0;
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
  uint c_bool = g.add_constant(true);
  uint c_real = g.add_constant(-2.5);
  uint c_natural_1 = g.add_constant((graph::natural_t)1);
  uint c_natural_2 = g.add_constant((graph::natural_t)2);
  uint c_prob = g.add_constant_probability(0.5);
  uint c_pos = g.add_constant_pos_real(2.5);
  uint c_neg = g.add_constant_neg_real(-1.5);

  Eigen::MatrixXd m0 = Eigen::MatrixXd::Constant(2, 1, 0.6);
  g.add_constant_probability_matrix(m0);
  Eigen::MatrixXd m1 = Eigen::MatrixXd::Identity(2, 2);
  g.add_constant_pos_matrix(m1);
  Eigen::MatrixXd m2 = Eigen::MatrixXd::Random(2, 2);
  g.add_constant_matrix(m2);
  Eigen::MatrixXd m3(2, 1);
  m3 << 0.2,
        0.8;
  g.add_constant_col_simplex_matrix(m3);
  Eigen::MatrixXb m4(1, 2);
  m4 << true, false;
  g.add_constant_matrix(m4);
  Eigen::MatrixXn m5(2, 1);
  m5 << 1,
        2;
  g.add_constant_matrix(m5);
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
  uint o_sample_bool = g.add_operator(
    graph::OperatorType::SAMPLE, std::vector<uint>{d_bernoulli});
  uint o_sample_real = g.add_operator(
    graph::OperatorType::SAMPLE, std::vector<uint>{d_normal});
  uint o_sample_natural = g.add_operator(
    graph::OperatorType::SAMPLE, std::vector<uint>{d_binomial});
  uint o_sample_prob = g.add_operator(
    graph::OperatorType::SAMPLE, std::vector<uint>{d_beta});
  uint o_sample_pos = g.add_operator(
    graph::OperatorType::SAMPLE, std::vector<uint>{d_gamma});

  uint o_iidsample_bool = g.add_operator(
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
  uint o_negate = g.add_operator(
    graph::OperatorType::NEGATE, std::vector<uint>{c_real});
  uint o_exp = g.add_operator(
    graph::OperatorType::EXP, std::vector<uint>{c_real});
  uint o_expm1 = g.add_operator(
    graph::OperatorType::EXPM1, std::vector<uint>{o_sample_pos});
  uint o_log = g.add_operator(
    graph::OperatorType::LOG, std::vector<uint> {o_to_pos});
  uint o_log1pexp = g.add_operator(
    graph::OperatorType::LOG1PEXP, std::vector<uint> {o_log});
  uint o_log1mexp = g.add_operator(
    graph::OperatorType::LOG1MEXP, std::vector<uint> {c_neg});
  uint o_logsumexp = g.add_operator(
    graph::OperatorType::LOGSUMEXP, std::vector<uint> {c_real, o_sample_real});
  uint o_multiply = g.add_operator(
    graph::OperatorType::MULTIPLY, std::vector<uint>{c_real, o_sample_real, o_logsumexp});
  uint o_add = g.add_operator(
    graph::OperatorType::ADD, std::vector<uint>{c_real, o_sample_real, o_to_real});
  uint o_phi = g.add_operator(
    graph::OperatorType::PHI, std::vector<uint>{o_sample_real});
  uint o_logistic = g.add_operator(
    graph::OperatorType::LOGISTIC, std::vector<uint>{c_real});
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
