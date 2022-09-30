/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include <gtest/gtest.h>

#include <beanmachine/graph/graph.h>
#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/factor/factor.h"
#include "beanmachine/graph/operator/operator.h"
#include "beanmachine/graph/util.h"

using namespace beanmachine;
using namespace std;
using namespace graph;
using namespace util;

void populate_arithmetic_graph(unique_ptr<Graph>& g) {
  /*
   o1 ~ Bernoulli(0.1)
   o2 = (positive real) o1
   o3 = 0.8 * o2
   o4 = 0.1 + o3
   o5 ~ Noisy_or(o4)

   or more simply:

   o1 ~ Bernoulli(0.1)
   o5 ~ Noisy_or(0.1 + 0.8 * o1)
   Observe o5 to be true
   Query o1.
  */
  uint c1 = g->add_constant_probability(0.1);
  uint d1 = g->add_distribution(
      DistributionType::BERNOULLI, AtomicType::BOOLEAN, vector<uint>({c1}));
  uint o1 = g->add_operator(OperatorType::SAMPLE, vector<uint>({d1}));
  uint o2 = g->add_operator(OperatorType::TO_POS_REAL, vector<uint>({o1}));
  uint c2 = g->add_constant_pos_real(0.8);
  uint o3 = g->add_operator(OperatorType::MULTIPLY, vector<uint>({c2, o2}));
  uint c3 = g->add_constant_pos_real(0.1);
  uint o4 = g->add_operator(OperatorType::ADD, vector<uint>({c3, o3}));
  uint d2 = g->add_distribution(
      DistributionType::BERNOULLI_NOISY_OR,
      AtomicType::BOOLEAN,
      vector<uint>({o4}));
  uint o5 = g->add_operator(OperatorType::SAMPLE, vector<uint>({d2}));
  // P(o5|o1=T) = 1 - exp(-.9)=0.5934 and P(o5|o1=F) = 1-exp(-.1)=0.09516
  // Since P(o1=T)=0.1 and P(o1=F)=0.9. Therefore P(o5=T,o1=T) = 0.05934,
  // P(o5=T,o1=F) = 0.08564 and P(o1=T | o5=T) = 0.4093
  g->observe(o5, true);
  g->query(o1);
}

unique_ptr<Graph> make_arithmetic_graph() {
  unique_ptr<Graph> g = make_unique<Graph>();
  populate_arithmetic_graph(g);
  return g;
}

void test_arithmetic_network(unique_ptr<Graph>& g) {
  vector<vector<NodeValue>> samples = g->infer(100, InferenceType::GIBBS);
  int sum = 0;
  for (const auto& sample : samples) {
    const auto& s = sample.front();
    ASSERT_EQ(s.type, AtomicType::BOOLEAN);
    sum += s._bool ? 1 : 0;
  }
  // less than 1 in million odds of failing these tests with correct infer
  EXPECT_LT(sum, 65);
  EXPECT_GT(sum, 15);
  // infer_mean should give almost exactly the same answer
  vector<double> means = g->infer_mean(100, InferenceType::GIBBS);
  EXPECT_TRUE(abs(sum - int(means[0] * 100)) <= 1);
  // repeat the test with rejection sampling
  vector<vector<NodeValue>> samples2 = g->infer(100, InferenceType::REJECTION);
  sum = 0;
  for (const auto& sample : samples2) {
    const auto& s = sample.front();
    ASSERT_EQ(s.type, AtomicType::BOOLEAN);
    sum += s._bool ? 1 : 0;
  }
  EXPECT_LT(sum, 65);
  EXPECT_GT(sum, 15);
  // infer_mean should give the same answer
  vector<double> means2 = g->infer_mean(100, InferenceType::REJECTION);
  EXPECT_TRUE(abs(sum - int(means2[0] * 100)) <= 1);
}

TEST(testgraph, infer_arithmetic) {
  unique_ptr<Graph> g = make_arithmetic_graph();
  test_arithmetic_network(g);
}

TEST(testgraph, remove_node) {
  unique_ptr<Graph> g = make_arithmetic_graph();

  size_t original_number_of_nodes = g->nodes.size();

  // Let's recover some of the nodes since we know where they are
  uint o1 = 2;
  uint o5 = g->nodes.size() - 1;
  uint d2 = g->nodes.size() - 2;
  uint o4 = g->nodes.size() - 3;

  // Let's make some new ones that are topologically later
  uint c4 = g->add_constant(0.7);
  uint o6 = g->add_operator(OperatorType::TO_REAL, vector<uint>({o1}));
  uint o7 = g->add_operator(OperatorType::MULTIPLY, vector<uint>({c4, o6}));
  uint o8 = g->add_operator(OperatorType::TO_REAL, vector<uint>({o5}));
  uint o9 = g->add_operator(OperatorType::MULTIPLY, vector<uint>({c4, o8}));

  // Cannot remove a node with out-nodes
  ASSERT_THROW(g->remove_node(c4), invalid_argument);

  // Need to remove them in reverse topological order
  // (that is, only nodes without out-nodes)

  // Ok to remove o7 and o6 first even though o8 and o9 were
  // added later, because o8 and o9 do not depend on o6 and o7
  auto from_old_to_new_id1 = g->remove_node(o7);
  auto from_old_to_new_id2 = g->remove_node(o6);

  // Cannot look up out-of-bounds old id
  auto out_of_bounds_id = o9 + 1;
  ASSERT_THROW(
      from_old_to_new_id2(from_old_to_new_id1(out_of_bounds_id)),
      invalid_argument);

  // However, removing o6 and o7 first means g->nodes got compacted
  // and since o8 and o9 had greater indices than o6 and o7,
  // they have new ids:
  o8 = from_old_to_new_id2(from_old_to_new_id1(o8));
  o9 = from_old_to_new_id2(from_old_to_new_id1(o9));

  // c4 did not change
  ASSERT_EQ(from_old_to_new_id2(from_old_to_new_id1(c4)), c4);

  // Let's remove all new nodes
  g->remove_node(o9);
  g->remove_node(o8);
  g->remove_node(c4);

  // Now are are back to the original graph
  test_arithmetic_network(g);
  ASSERT_EQ(g->nodes.size(), original_number_of_nodes);

  // Now let's remove all nodes and put them back to see if nothing breaks.
  auto number_of_remaining_nodes = original_number_of_nodes;
  while (number_of_remaining_nodes != 0) {
    auto last_node_id = number_of_remaining_nodes - 1;
    g->remove_node(last_node_id);
    number_of_remaining_nodes--;
  }
  populate_arithmetic_graph(g);
  test_arithmetic_network(g);
}

TEST(testgraph, infer_bn) {
  Graph g;
  // classic sprinkler BN, see the diagram here:
  // https://upload.wikimedia.org/wikipedia/commons/0/0e/SimpleBayesNet.svg
  Eigen::MatrixXd matrix1(2, 1);
  matrix1 << 0.8, 0.2;
  uint c1 = g.add_constant_col_simplex_matrix(matrix1);
  uint d1 = g.add_distribution(
      DistributionType::TABULAR, AtomicType::BOOLEAN, vector<uint>({c1}));
  uint RAIN = g.add_operator(OperatorType::SAMPLE, vector<uint>({d1}));
  Eigen::MatrixXd matrix2(2, 2);
  matrix2 << 0.6, 0.99, 0.4, 0.01;
  uint c2 = g.add_constant_col_simplex_matrix(matrix2);
  uint d2 = g.add_distribution(
      DistributionType::TABULAR, AtomicType::BOOLEAN, vector<uint>({c2, RAIN}));
  uint SPRINKLER = g.add_operator(OperatorType::SAMPLE, vector<uint>({d2}));
  Eigen::MatrixXd matrix3(2, 4);
  matrix3 << 1.0, 0.2, 0.1, 0.01, 0.0, 0.8, 0.9, 0.99;
  uint c3 = g.add_constant_col_simplex_matrix(matrix3);
  uint d3 = g.add_distribution(
      DistributionType::TABULAR,
      AtomicType::BOOLEAN,
      vector<uint>({c3, SPRINKLER, RAIN}));
  uint GRASSWET = g.add_operator(OperatorType::SAMPLE, vector<uint>({d3}));
  g.observe(GRASSWET, true);
  g.query(RAIN);
  uint n_iter = 100;
  const vector<vector<NodeValue>>& samples =
      g.infer(n_iter, InferenceType::REJECTION);
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
  const auto& all_samples = g.infer(n_iter, InferenceType::REJECTION, 123, 2);
  ASSERT_EQ(all_samples.size(), 2);
  ASSERT_EQ(all_samples[0].size(), n_iter);
  ASSERT_EQ(all_samples[1].size(), n_iter);

  uint eqsum = 0;
  for (int i = 0; i < n_iter; i++) {
    const auto& s0 = all_samples[0][i].front();
    const auto& s1 = all_samples[1][i].front();
    // @lint-ignore CLANGTIDY
    eqsum += (s0._bool == s1._bool) ? 1 : 0;
  }
  ASSERT_LT(eqsum, n_iter);

  const auto& all_means =
      g.infer_mean(n_iter, InferenceType::REJECTION, 123, 2);
  ASSERT_EQ(all_means.size(), 2);
  ASSERT_NE(all_means[0].front(), all_means[1].front());
  ASSERT_GT(all_means[0].front(), 0.1);
  ASSERT_LT(all_means[0].front(), 0.6);
  ASSERT_GT(all_means[1].front(), 0.1);
  ASSERT_LT(all_means[1].front(), 0.6);
}

/*
  A graph with nodes of all types. It is not a meaningful model.
  It is designed to test Graph's copy constructor and Node's clone.
*/
unique_ptr<Graph> make_graph_with_nodes_of_all_types() {
  auto g = make_unique<Graph>();

  // constants
  uint c_bool = g->add_constant_bool(true);
  uint c_real = g->add_constant_real(-2.5);
  uint c_natural_1 = g->add_constant_natural(1);
  uint c_natural_2 = g->add_constant_natural(2);
  uint c_prob = g->add_constant_probability(0.5);
  uint c_pos = g->add_constant_pos_real(2.5);
  uint c_neg = g->add_constant_neg_real(-1.5);

  Eigen::MatrixXd m0 = Eigen::MatrixXd::Constant(2, 1, 0.6);
  uint cm0 = g->add_constant_probability_matrix(m0);
  Eigen::MatrixXd m1 = Eigen::MatrixXd::Identity(2, 2);
  uint cm1 = g->add_constant_pos_matrix(m1);
  Eigen::MatrixXd m2 = Eigen::MatrixXd::Random(2, 2);
  uint cm2 = g->add_constant_real_matrix(m2);
  Eigen::MatrixXd m3(2, 1);
  m3 << 0.2, 0.8;
  g->add_constant_col_simplex_matrix(m3);
  Eigen::MatrixXb m4(1, 2);
  m4 << true, false;
  g->add_constant_bool_matrix(m4);
  Eigen::MatrixXn m5(2, 1);
  m5 << 1, 2;
  g->add_constant_natural_matrix(m5);

  Eigen::MatrixXd matrix1(2, 1);
  matrix1 << 0.8, 0.2;
  uint c1 = g->add_constant_col_simplex_matrix(matrix1);

  Eigen::MatrixXd matrix2(3, 1);
  matrix2 << 10.0, 20.00, 30.00;
  uint c2 = g->add_constant_pos_matrix(matrix2);

  const double LOG_MEAN = -11.0;
  const double LOG_STD = 3.0;
  auto real1 = g->add_constant(LOG_MEAN);
  auto real2 = g->add_constant_real(0.5);
  auto real3 = g->add_constant_real(3.0);
  auto pos1 = g->add_constant_pos_real(LOG_STD);
  auto pos2 = g->add_constant_pos_real(2.5);

  auto prob1 = g->add_constant_probability(0.3);

  auto zero = g->add_constant_real(0.0);
  auto zero_nat = g->add_constant_natural(0);
  auto one = g->add_constant_natural(1);
  auto two = g->add_constant_natural(2);
  auto three = g->add_constant_natural(3);

  auto pos_zero = g->add_constant_pos_real(0.0);
  auto loc = g->add_operator(OperatorType::ADD, vector<uint>{zero, real1});
  auto scale = g->add_operator(OperatorType::ADD, vector<uint>{pos_zero, pos1});

  // distributions
  auto lkj_chol_dist = g->add_distribution(
      DistributionType::LKJ_CHOLESKY,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 5, 5),
      vector<uint>{pos1});

  auto cauchy_dist = g->add_distribution(
      DistributionType::CAUCHY, AtomicType::REAL, vector<uint>{loc, scale});

  auto geometric_dist = g->add_distribution(
      DistributionType::GEOMETRIC, AtomicType::NATURAL, vector<uint>{prob1});

  auto poisson_dist = g->add_distribution(
      DistributionType::POISSON, AtomicType::NATURAL, vector<uint>{pos1});

  uint categorical_dist = g->add_distribution(
      DistributionType::CATEGORICAL, AtomicType::NATURAL, vector<uint>({c1}));

  auto normal_dist1 = g->add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, vector<uint>{real1, pos1});

  auto normal_dist2 = g->add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, vector<uint>{real2, pos2});

  auto product_dist = g->add_distribution(
      DistributionType::PRODUCT,
      AtomicType::UNKNOWN, // OK -- parents will determine
      vector<uint>{normal_dist1, normal_dist2});

  auto bimix_dist = g->add_distribution(
      DistributionType::BIMIXTURE,
      AtomicType::REAL,
      vector<uint>{prob1, normal_dist1, normal_dist2});

  uint tabular_dist = g->add_distribution(
      DistributionType::TABULAR, AtomicType::BOOLEAN, vector<uint>({c1}));

  uint d_bernoulli = g->add_distribution(
      DistributionType::BERNOULLI, AtomicType::BOOLEAN, vector<uint>{c_prob});

  uint flat_dist = g->add_distribution(
      DistributionType::FLAT,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::POS_REAL, 3, 1),
      vector<uint>{});
  uint flat_sample = g->add_operator(OperatorType::SAMPLE, {flat_dist});

  uint diri_dist = g->add_distribution(
      DistributionType::DIRICHLET,
      ValueType(
          VariableType::COL_SIMPLEX_MATRIX, AtomicType::PROBABILITY, 3, 1),
      vector<uint>{flat_sample});

  auto log_normal_dist = g->add_distribution(
      DistributionType::LOG_NORMAL,
      AtomicType::POS_REAL,
      vector<uint>{real1, pos1});

  auto half_normal_dist = g->add_distribution(
      DistributionType::HALF_NORMAL, AtomicType::POS_REAL, vector<uint>{pos1});

  uint flat_dist_mean = g->add_distribution(
      DistributionType::FLAT,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 1),
      vector<uint>{});
  uint flat_sample_mean =
      g->add_operator(OperatorType::SAMPLE, {flat_dist_mean});

  uint flat_dist_cov = g->add_distribution(
      DistributionType::FLAT,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 3),
      vector<uint>{});
  uint flat_sample_cov = g->add_operator(OperatorType::SAMPLE, {flat_dist_cov});

  Eigen::MatrixXd m6(3, 1);
  m6 << 1.5, 1.0, 2.0;

  Eigen::MatrixXd m7(3, 3);
  m7 << 1.0, 0.5, 0.0, 0.5, 1.0, 0.25, 0.0, 0.25, 1.0;

  g->observe(flat_sample_mean, m6);
  g->observe(flat_sample_cov, m7);

  uint multivariate_dist = g->add_distribution(
      DistributionType::MULTIVARIATE_NORMAL,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 1),
      vector<uint>{flat_sample_mean, flat_sample_cov});

  uint d_bernoulli_or = g->add_distribution(
      DistributionType::BERNOULLI_NOISY_OR,
      AtomicType::BOOLEAN,
      vector<uint>{c_pos});
  uint d_bernoulli_logit = g->add_distribution(
      DistributionType::BERNOULLI_LOGIT,
      AtomicType::BOOLEAN,
      vector<uint>{c_real});
  uint d_beta = g->add_distribution(
      DistributionType::BETA,
      AtomicType::PROBABILITY,
      vector<uint>{c_pos, c_pos});
  uint d_binomial = g->add_distribution(
      DistributionType::BINOMIAL,
      AtomicType::NATURAL,
      vector<uint>{c_natural_2, c_prob});
  uint d_flat = g->add_distribution(
      DistributionType::FLAT, AtomicType::POS_REAL, vector<uint>{});
  uint d_normal = g->add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, vector<uint>{c_real, c_pos});
  uint d_halfcauchy = g->add_distribution(
      DistributionType::HALF_CAUCHY, AtomicType::POS_REAL, vector<uint>{c_pos});
  uint d_studentt = g->add_distribution(
      DistributionType::STUDENT_T,
      AtomicType::REAL,
      vector<uint>{c_pos, c_real, c_pos});
  uint d_gamma = g->add_distribution(
      DistributionType::GAMMA,
      AtomicType::POS_REAL,
      vector<uint>{c_pos, c_pos});

  // operators
  uint o_sample_bool =
      g->add_operator(OperatorType::SAMPLE, vector<uint>{d_bernoulli});
  uint o_sample_real =
      g->add_operator(OperatorType::SAMPLE, vector<uint>{d_normal});
  uint o_sample_natural =
      g->add_operator(OperatorType::SAMPLE, vector<uint>{d_binomial});
  uint o_sample_prob =
      g->add_operator(OperatorType::SAMPLE, vector<uint>{d_beta});
  uint o_sample_pos =
      g->add_operator(OperatorType::SAMPLE, vector<uint>{d_gamma});

  uint o_iidsample_bool = g->add_operator(
      OperatorType::IID_SAMPLE,
      vector<uint>{d_bernoulli, c_natural_1, c_natural_2});
  uint o_iidsample_real = g->add_operator(
      OperatorType::IID_SAMPLE,
      vector<uint>{d_normal, c_natural_2, c_natural_2});
  uint o_iidsample_natural = g->add_operator(
      OperatorType::IID_SAMPLE, vector<uint>{d_binomial, c_natural_2});
  uint o_iidsample_prob = g->add_operator(
      OperatorType::IID_SAMPLE, vector<uint>{d_beta, c_natural_2, c_natural_1});
  uint o_iidsample_pos = g->add_operator(
      OperatorType::IID_SAMPLE,
      vector<uint>{d_gamma, c_natural_2, c_natural_2});

  uint o_to_real =
      g->add_operator(OperatorType::TO_REAL, vector<uint>{o_sample_pos});
  uint o_to_pos =
      g->add_operator(OperatorType::TO_POS_REAL, vector<uint>{o_sample_prob});
  uint o_complement =
      g->add_operator(OperatorType::COMPLEMENT, vector<uint>{o_sample_prob});
  uint o_negate = g->add_operator(OperatorType::NEGATE, vector<uint>{c_real});
  uint o_exp = g->add_operator(OperatorType::EXP, vector<uint>{c_real});
  uint o_expm1 =
      g->add_operator(OperatorType::EXPM1, vector<uint>{o_sample_pos});
  uint o_log = g->add_operator(OperatorType::LOG, vector<uint>{o_to_pos});
  uint o_log1pexp =
      g->add_operator(OperatorType::LOG1PEXP, vector<uint>{o_log});
  uint o_log1mexp =
      g->add_operator(OperatorType::LOG1MEXP, vector<uint>{c_neg});
  uint o_logsumexp = g->add_operator(
      OperatorType::LOGSUMEXP, vector<uint>{c_real, o_sample_real});
  uint o_multiply = g->add_operator(
      OperatorType::MULTIPLY, vector<uint>{c_real, o_sample_real, o_logsumexp});
  uint o_add = g->add_operator(
      OperatorType::ADD, vector<uint>{c_real, o_sample_real, o_to_real});
  uint o_phi = g->add_operator(OperatorType::PHI, vector<uint>{o_sample_real});
  uint o_logistic =
      g->add_operator(OperatorType::LOGISTIC, vector<uint>{c_real});
  uint o_ifelse = g->add_operator(
      OperatorType::IF_THEN_ELSE,
      vector<uint>{o_sample_bool, o_sample_pos, o_log1pexp});

  auto x = g->add_operator(
      OperatorType::IID_SAMPLE, vector<uint>{normal_dist1, three, two});
  auto y = g->add_operator(
      OperatorType::IID_SAMPLE, vector<uint>{normal_dist1, three, two});
  Eigen::MatrixXd m8(3, 2);
  m8 << 0.3, -0.1, 1.2, 0.9, -2.6, 0.8;
  Eigen::MatrixXd m9(3, 2);
  m9 << 0.4, 0.1, 0.5, -1.1, 0.7, -0.6;
  g->observe(x, m8);
  g->observe(y, m9);
  auto xy =
      g->add_operator(OperatorType::ELEMENTWISE_MULTIPLY, vector<uint>{x, y});

  auto neg = g->add_operator(OperatorType::MATRIX_NEGATE, vector<uint>{cm2});

  auto neg1 = g->add_constant_neg_real(-1.0);
  auto pospos =
      g->add_operator(OperatorType::TO_MATRIX, {two, one, pos1, pos1});
  auto negneg =
      g->add_operator(OperatorType::TO_MATRIX, {two, one, neg1, neg1});
  g->add_operator(OperatorType::LOGSUMEXP_VECTOR, vector<uint>{pospos});
  g->add_operator(OperatorType::LOGSUMEXP_VECTOR, vector<uint>{negneg});

  auto prob1_pow_pos2 =
      g->add_operator(OperatorType::POW, vector<uint>{prob1, pos2});

  uint cm1t = g->add_operator(OperatorType::TRANSPOSE, vector<uint>{cm1});

  auto cm1_x_cm2 =
      g->add_operator(OperatorType::MATRIX_MULTIPLY, vector<uint>{cm1, cm2});

  auto real1_cm2_scale =
      g->add_operator(OperatorType::MATRIX_SCALE, vector<uint>{real1, cm2});

  auto cm2_cm2_add =
      g->add_operator(OperatorType::MATRIX_ADD, vector<uint>{cm2, cm2});

  auto prob2 =
      g->add_operator(OperatorType::TO_PROBABILITY, vector<uint>{real1});

  uint first_element =
      g->add_operator(OperatorType::INDEX, vector<uint>{c1, zero_nat});

  uint first_column =
      g->add_operator(OperatorType::COLUMN_INDEX, vector<uint>{cm2, zero_nat});

  uint sum_matrix =
      g->add_operator(OperatorType::BROADCAST_ADD, vector<uint>{real1, cm2});

  uint tm = g->add_operator(
      OperatorType::TO_MATRIX,
      vector<uint>{two, two, prob1, prob1, prob1, prob1});
  uint real_matrix =
      g->add_operator(OperatorType::TO_REAL_MATRIX, vector<uint>{tm});

  uint pos_real_matrix =
      g->add_operator(OperatorType::TO_POS_REAL_MATRIX, vector<uint>{tm});

  uint neg_real =
      g->add_operator(OperatorType::TO_NEG_REAL, vector<uint>{real1});

  uint choice = g->add_operator(
      OperatorType::CHOICE, vector<uint>{zero_nat, real1, real2, real3});

  uint integer = g->add_operator(OperatorType::TO_INT, vector<uint>{real1});

  Eigen::MatrixXd positive_definite(3, 3);
  positive_definite << 10, 5, 2, 5, 3, 2, 2, 2, 3;
  auto positive_definite_matrix =
      g->add_constant_real_matrix(positive_definite);
  auto l = g->add_operator(OperatorType::CHOLESKY, {positive_definite_matrix});

  auto exp = g->add_operator(OperatorType::MATRIX_EXP, {cm1});

  auto log_prob =
      g->add_operator(OperatorType::LOG_PROB, {normal_dist1, real1});

  auto matrix_sum = g->add_operator(OperatorType::MATRIX_SUM, {cm1});

  auto matrix_log = g->add_operator(OperatorType::MATRIX_LOG, {cm1});

  uint log1p = g->add_operator(OperatorType::LOG1P, vector<uint>{real1});

  auto mlog1p = g->add_operator(OperatorType::MATRIX_LOG1P, {cm1});

  Eigen::MatrixXd neg_m1(3, 1);
  neg_m1 << -2.0, -1.0, -3.0;
  auto neg_cm1 = g->add_constant_neg_matrix(neg_m1);
  auto mlog1mexp = g->add_operator(OperatorType::MATRIX_LOG1MEXP, {neg_cm1});

  auto mphi = g->add_operator(OperatorType::MATRIX_PHI, {cm2});

  Eigen::MatrixXd probs(2, 1);
  probs << 0.2, 0.7;
  auto probs_matrix = g->add_constant_probability_matrix(probs);
  auto matrix_complement =
      g->add_operator(OperatorType::MATRIX_COMPLEMENT, {probs_matrix});

  auto fill_matrix = g->add_operator(
      OperatorType::FILL_MATRIX,
      vector<uint>{neg_real, c_natural_1, c_natural_2});

  auto broadcast = g->add_operator(
      OperatorType::BROADCAST,
      vector<uint>{fill_matrix, c_natural_2, c_natural_2});

  // factors
  uint f_expprod = g->add_factor(
      FactorType::EXP_PRODUCT,
      vector<uint>{o_sample_real, o_sample_pos, o_to_pos});
  // observe and query
  g->observe(o_sample_real, 1.5);
  g->observe(o_sample_prob, 0.1);

  g->observe(o_iidsample_prob, m0);
  g->observe(o_iidsample_pos, m1);
  g->observe(o_iidsample_real, m2);
  g->observe(o_iidsample_bool, m4);
  g->observe(o_iidsample_natural, m5);

  g->query(o_multiply);
  g->query(o_add);
  g->query(o_ifelse);
  g->query(o_log1mexp);

  return g;
}

TEST(testgraph, graph_with_nodes_of_all_types_indeed_has_them_all) {
  using namespace distribution;
  using namespace oper;
  using namespace factor;

  auto g = make_graph_with_nodes_of_all_types();

  /// Distributions

  set<DistributionType> missing_distributions;
  for (auto distribution_type : DistributionTypeIterable()) {
    auto is_distribution_of_this_type = [distribution_type](auto& node) {
      return node->node_type == NodeType::DISTRIBUTION and
          dynamic_cast<Distribution*>(node.get())->dist_type ==
          distribution_type;
    };

    if (not any_of(
            g->nodes.begin(), g->nodes.end(), is_distribution_of_this_type)) {
      missing_distributions.insert(distribution_type);
    }
  }

  missing_distributions.erase(DistributionType::DUMMY); // soon to be discarded

  if (not missing_distributions.empty()) {
    cerr << "Graph does not contain distributions of types ";
    for (auto type : missing_distributions) {
      cerr << NAMEOF_ENUM(type) << ", ";
    }
    FAIL();
  }

  /// Operators

  set<OperatorType> missing_operators;
  for (auto operator_type : OperatorTypeIterable()) {
    auto is_operator_of_this_type = [operator_type](auto& node) {
      return node->node_type == NodeType::OPERATOR and
          dynamic_cast<Operator*>(node.get())->op_type == operator_type;
    };

    if (not any_of(
            g->nodes.begin(), g->nodes.end(), is_operator_of_this_type)) {
      missing_operators.insert(operator_type);
    }
  }

  if (not missing_operators.empty()) {
    cerr << "Graph does not contain operators of types ";
    for (auto type : missing_operators) {
      cerr << NAMEOF_ENUM(type) << ", ";
    }
    FAIL();
  }

  /// Factors

  set<FactorType> missing_factors;
  for (auto factor_type : FactorTypeIterable()) {
    auto is_factor_of_this_type = [factor_type](auto& node) {
      return node->node_type == NodeType::FACTOR and
          dynamic_cast<Factor*>(node.get())->fac_type == factor_type;
    };

    if (not any_of(g->nodes.begin(), g->nodes.end(), is_factor_of_this_type)) {
      missing_factors.insert(factor_type);
    }
  }

  if (not missing_factors.empty()) {
    cerr << "Graph does not contain factors of types ";
    for (auto type : missing_factors) {
      cerr << NAMEOF_ENUM(type) << ", ";
    }
    FAIL();
  }
}

TEST(testgraph, graph_copy_constructor) {
  auto g = make_graph_with_nodes_of_all_types();
  Graph g_copy(*g);
  ASSERT_EQ(g->to_string(), g_copy.to_string());
}

TEST(testgraph, test_node_to_string) {
  auto g = make_graph_with_nodes_of_all_types();
  stringstream strstr;
  for (auto& node : g->nodes) {
    strstr << node->to_string() << endl;
  }
  ASSERT_EQ(
      strstr.str(),
      R"(CONSTANT(boolean 1)
CONSTANT(real -2.5)
CONSTANT(natural 1)
CONSTANT(natural 2)
CONSTANT(probability 0.5)
CONSTANT(positive real 2.5)
CONSTANT(negative real -1.5)
CONSTANT(matrix<probability> 0.6
0.6)
CONSTANT(matrix<positive real> 1 0
0 1)
CONSTANT(matrix<real>  0.680375  0.566198
-0.211234   0.59688)
CONSTANT(col_simplex_matrix<probability> 0.2
0.8)
CONSTANT(matrix<boolean> 1 0)
CONSTANT(matrix<natural> 1
2)
CONSTANT(col_simplex_matrix<probability> 0.8
0.2)
CONSTANT(matrix<positive real> 10
20
30)
CONSTANT(real -11)
CONSTANT(real 0.5)
CONSTANT(real 3)
CONSTANT(positive real 3)
CONSTANT(positive real 2.5)
CONSTANT(probability 0.3)
CONSTANT(real 0)
CONSTANT(natural 0)
CONSTANT(natural 1)
CONSTANT(natural 2)
CONSTANT(natural 3)
CONSTANT(positive real 1e-10)
ADD(21, 15)
ADD(26, 18)
LKJ_CHOLESKY(18)
CAUCHY(27, 28)
GEOMETRIC(20)
POISSON(18)
CATEGORICAL(13)
NORMAL(15, 18)
NORMAL(16, 19)
PRODUCT(34, 35)
BIMIXTURE(20, 34, 35)
TABULAR(13)
BERNOULLI(4)
FLAT()
SAMPLE(40)
DIRICHLET(41)
LOG_NORMAL(15, 18)
HALF_NORMAL(18)
FLAT()
SAMPLE(45)
FLAT()
SAMPLE(47)
MULTIVARIATE_NORMAL(46, 48)
BERNOULLI_NOISY_OR(5)
BERNOULLI_LOGIT(1)
BETA(5, 5)
BINOMIAL(3, 4)
FLAT()
NORMAL(1, 5)
HALF_CAUCHY(5)
STUDENT_T(5, 1, 5)
GAMMA(5, 5)
SAMPLE(39)
SAMPLE(55)
SAMPLE(53)
SAMPLE(52)
SAMPLE(58)
IID_SAMPLE(39, 2, 3)
IID_SAMPLE(55, 3, 3)
IID_SAMPLE(53, 3)
IID_SAMPLE(52, 3, 2)
IID_SAMPLE(58, 3, 3)
TO_REAL(63)
TO_POS_REAL(62)
COMPLEMENT(62)
NEGATE(1)
EXP(1)
EXPM1(63)
LOG(70)
LOG1PEXP(75)
LOG1MEXP(6)
LOGSUMEXP(1, 60)
MULTIPLY(1, 60, 78)
ADD(1, 60, 69)
PHI(60)
LOGISTIC(1)
IF_THEN_ELSE(59, 63, 76)
IID_SAMPLE(34, 25, 24)
IID_SAMPLE(34, 25, 24)
ELEMENTWISE_MULTIPLY(84, 85)
MATRIX_NEGATE(9)
CONSTANT(negative real -1)
TO_MATRIX(24, 23, 18, 18)
TO_MATRIX(24, 23, 88, 88)
LOGSUMEXP_VECTOR(89)
LOGSUMEXP_VECTOR(90)
POW(20, 19)
TRANSPOSE(8)
MATRIX_MULTIPLY(8, 9)
MATRIX_SCALE(15, 9)
MATRIX_ADD(9, 9)
TO_PROBABILITY(15)
INDEX(13, 22)
COLUMN_INDEX(9, 22)
BROADCAST_ADD(15, 9)
TO_MATRIX(24, 24, 20, 20, 20, 20)
TO_REAL_MATRIX(102)
TO_POS_REAL_MATRIX(102)
TO_NEG_REAL(15)
CHOICE(22, 15, 16, 17)
TO_INT(15)
CONSTANT(matrix<real> 10  5  2
 5  3  2
 2  2  3)
CHOLESKY(108)
MATRIX_EXP(8)
LOG_PROB(34, 15)
MATRIX_SUM(8)
MATRIX_LOG(8)
LOG1P(15)
MATRIX_LOG1P(8)
CONSTANT(matrix<negative real> -2
-1
-3)
MATRIX_LOG1MEXP(116)
MATRIX_PHI(9)
CONSTANT(matrix<probability> 0.2
0.7)
MATRIX_COMPLEMENT(119)
FILL_MATRIX(105, 2, 3)
BROADCAST(121, 3, 3)
EXP_PRODUCT(60, 63, 70)
)");
}

TEST(testgraph, test_node_cloning) {
  auto g = make_graph_with_nodes_of_all_types();

  Graph original_g_copy(*g);

  // Duplicate all nodes
  auto original_size = g->nodes.size();
  for (auto node_id : range(original_size)) {
    auto& node = g->nodes[node_id];
    uint clone_id = g->duplicate(node);
  }

  for (auto node_id : range(original_size)) {
    auto& original_node = g->nodes[node_id];
    auto& clone_node = g->nodes[node_id + original_size];
    ASSERT_TRUE(are_equal(*original_node, *clone_node));
  }
}

TEST(testgraph, full_log_prob) {
  Graph g;
  uint a = g.add_constant_pos_real(5.0);
  uint b = g.add_constant_pos_real(3.0);
  uint n = g.add_constant_natural(10);
  uint beta = g.add_distribution(
      DistributionType::BETA, AtomicType::PROBABILITY, vector<uint>({a, b}));
  uint prob = g.add_operator(OperatorType::SAMPLE, vector<uint>({beta}));
  uint binomial = g.add_distribution(
      DistributionType::BINOMIAL, AtomicType::NATURAL, vector<uint>({n, prob}));
  uint k = g.add_operator(OperatorType::SAMPLE, vector<uint>({binomial}));
  g.observe(k, (natural_t)8);
  g.query(prob);
  // Note: the posterior is p ~ Beta(13, 5).
  int num_samples = 10;
  auto conf = InferConfig(true);
  auto samples = g.infer(num_samples, InferenceType::NMC, 123, 2, conf);
  auto log_probs = g.get_log_prob();
  EXPECT_EQ(log_probs.size(), 2);
  EXPECT_EQ(log_probs[0].size(), num_samples);
  for (uint c = 0; c < 2; ++c) {
    for (uint i = 0; i < num_samples; ++i) {
      auto& s = samples[c][i][0];
      double l = log_probs[c][i];
      g.remove_observations();
      g.observe(k, (natural_t)8);
      g.observe(prob, s._double);
      EXPECT_NEAR(g.full_log_prob(), l, 1e-3);
    }
  }
  g.remove_observations();
  g.observe(k, (natural_t)8);
  g.observe(prob, 0.6);
  EXPECT_NEAR(g.full_log_prob(), -1.3344, 1e-3);
}

TEST(testgraph, bad_observations) {
  // Tests which demonstrate that we give errors for bad observations.
  Eigen::MatrixXb bool_matrix(1, 2);
  bool_matrix << true, false;
  Eigen::MatrixXn nat_matrix(1, 2);
  nat_matrix << 2, 3;
  Eigen::MatrixXd real_matrix(1, 2);
  real_matrix << 1.5, 2.5;
  natural_t nat = 2;

  Graph g;

  // Observe a bool to be a double, natural, matrix:
  uint c_prob = g.add_constant_probability(0.5);
  uint d_bernoulli = g.add_distribution(
      DistributionType::BERNOULLI, AtomicType::BOOLEAN, vector<uint>{c_prob});
  uint o_sample_bool =
      g.add_operator(OperatorType::SAMPLE, vector<uint>{d_bernoulli});
  EXPECT_THROW(g.observe(o_sample_bool, 0.1), invalid_argument);
  EXPECT_THROW(g.observe(o_sample_bool, nat), invalid_argument);
  EXPECT_THROW(g.observe(o_sample_bool, bool_matrix), invalid_argument);
  EXPECT_THROW(g.observe(o_sample_bool, nat_matrix), invalid_argument);
  EXPECT_THROW(g.observe(o_sample_bool, real_matrix), invalid_argument);

  // Observe a bool(2, 1) to be a bool, double, natural, and (1, 2) matrices
  uint c_natural_1 = g.add_constant_natural(1);
  uint c_natural_2 = g.add_constant_natural(2);
  uint o_iid_bool = g.add_operator(
      OperatorType::IID_SAMPLE,
      vector<uint>{d_bernoulli, c_natural_2, c_natural_1});
  EXPECT_THROW(g.observe(o_iid_bool, false), invalid_argument);
  EXPECT_THROW(g.observe(o_iid_bool, 0.1), invalid_argument);
  EXPECT_THROW(g.observe(o_iid_bool, nat), invalid_argument);
  EXPECT_THROW(g.observe(o_iid_bool, bool_matrix), invalid_argument);
  EXPECT_THROW(g.observe(o_iid_bool, nat_matrix), invalid_argument);
  EXPECT_THROW(g.observe(o_iid_bool, real_matrix), invalid_argument);

  // Observe a natural to be bool, real, matrix
  uint d_binomial = g.add_distribution(
      DistributionType::BINOMIAL,
      AtomicType::NATURAL,
      vector<uint>{c_natural_2, c_prob});
  uint o_sample_natural =
      g.add_operator(OperatorType::SAMPLE, vector<uint>{d_binomial});
  EXPECT_THROW(g.observe(o_sample_natural, true), invalid_argument);
  EXPECT_THROW(g.observe(o_sample_natural, 0.5), invalid_argument);
  EXPECT_THROW(g.observe(o_sample_natural, bool_matrix), invalid_argument);
  EXPECT_THROW(g.observe(o_sample_natural, nat_matrix), invalid_argument);
  EXPECT_THROW(g.observe(o_sample_natural, real_matrix), invalid_argument);

  // Observe a natural(2, 1) to be a bool, double, natural, and (1, 2)
  // matrices
  uint o_iid_nat = g.add_operator(
      OperatorType::IID_SAMPLE,
      vector<uint>{d_binomial, c_natural_2, c_natural_1});
  EXPECT_THROW(g.observe(o_iid_nat, false), invalid_argument);
  EXPECT_THROW(g.observe(o_iid_nat, 0.1), invalid_argument);
  EXPECT_THROW(g.observe(o_iid_nat, nat), invalid_argument);
  EXPECT_THROW(g.observe(o_iid_nat, bool_matrix), invalid_argument);
  EXPECT_THROW(g.observe(o_iid_nat, nat_matrix), invalid_argument);
  EXPECT_THROW(g.observe(o_iid_nat, real_matrix), invalid_argument);

  // Observe a real to be bool, natural, matrix:
  uint c_real = g.add_constant_real(-2.5);
  uint c_pos = g.add_constant_pos_real(2.5);
  uint d_normal = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, vector<uint>{c_real, c_pos});
  uint o_sample_real =
      g.add_operator(OperatorType::SAMPLE, vector<uint>{d_normal});
  EXPECT_THROW(g.observe(o_sample_real, true), invalid_argument);
  EXPECT_THROW(g.observe(o_sample_real, nat), invalid_argument);
  EXPECT_THROW(g.observe(o_sample_real, bool_matrix), invalid_argument);
  EXPECT_THROW(g.observe(o_sample_real, nat_matrix), invalid_argument);
  EXPECT_THROW(g.observe(o_sample_real, real_matrix), invalid_argument);

  // Observe a real(2, 1) to be a bool, double, natural, and (1, 2) matrices
  uint o_iid_real = g.add_operator(
      OperatorType::IID_SAMPLE,
      vector<uint>{d_normal, c_natural_2, c_natural_1});
  EXPECT_THROW(g.observe(o_iid_real, false), invalid_argument);
  EXPECT_THROW(g.observe(o_iid_real, 0.1), invalid_argument);
  EXPECT_THROW(g.observe(o_iid_real, nat), invalid_argument);
  EXPECT_THROW(g.observe(o_iid_real, bool_matrix), invalid_argument);
  EXPECT_THROW(g.observe(o_iid_real, nat_matrix), invalid_argument);
  EXPECT_THROW(g.observe(o_iid_real, real_matrix), invalid_argument);
}

TEST(testgraph, infer_runtime_error) {
  Graph g;
  auto two = g.add_constant_natural(2);
  Eigen::MatrixXd real_matrix(1, 1);
  real_matrix << 1.0;
  auto matrix = g.add_constant_real_matrix(real_matrix);
  // index out of bounds during runtime
  auto indexed_matrix = g.add_operator(OperatorType::INDEX, {matrix, two});
  g.query(indexed_matrix);

  int num_samples = 10;
  int seed = 19;
  // test with one chain
  EXPECT_THROW(g.infer(num_samples, InferenceType::NMC), runtime_error);
  // test with threads from multiple chains
  EXPECT_THROW(
      g.infer(num_samples, InferenceType::NMC, seed, 2), runtime_error);
}

TEST(testgraph, eval_and_update_backgrad) {
  /*
  PyTorch verification
  normal_dist = dist.Normal(0.0, 5.0)
  x = tensor(2.0, requires_grad=True)
  normal1_dist = dist.Normal(x * x, 10.0)
  y = tensor(100.0)
  log_prob = normal_dist.log_prob(x) + normal1_dist.log_prob(y)
  torch.autograd.grad(log_prob, x)
  */
  Graph g;
  auto mean0 = g.add_constant_real(0.0);
  auto sigma0 = g.add_constant_pos_real(5.0);
  auto dist0 = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, vector<uint>{mean0, sigma0});
  auto x = g.add_operator(OperatorType::SAMPLE, vector<uint>{dist0});
  auto x_sq = g.add_operator(OperatorType::MULTIPLY, vector<uint>{x, x});
  auto sigma1 = g.add_constant_pos_real(10.0);
  auto dist1 = g.add_distribution(
      DistributionType::NORMAL, AtomicType::REAL, vector<uint>{x_sq, sigma1});
  auto y = g.add_operator(OperatorType::SAMPLE, vector<uint>{dist1});
  g.observe(y, 100.0);
  g.query(x_sq);
  g.nodes[x]->value._double = 2.0;
  g.eval_and_update_backgrad(g.supp());
  EXPECT_NEAR(g.nodes[x]->back_grad1, 3.76, 1e-5);
}
