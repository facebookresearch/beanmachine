/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "beanmachine/graph/distribution/distribution.h"
#include "beanmachine/graph/global/nuts.h"
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/third-party/nameof.h"
#include "beanmachine/graph/util.h"

#include <boost/math/special_functions/factorials.hpp>
#include <math.h>
#include <chrono>

using namespace beanmachine::graph;
using namespace beanmachine::distribution;
using namespace beanmachine::util;
using namespace std;

TEST(testdistrib, product_construction_and_log_prob) {
  // This tests whether invalid arguments throw the right exceptions,
  // and that log_prob works as expected.
  Graph g;

  // Product must have at least one parent
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::PRODUCT, AtomicType::REAL, std::vector<uint>{}),
      std::invalid_argument);

  const double MEAN1 = -11.0;
  const double STD1 = 3.0;
  auto real1 = g.add_constant_real(MEAN1);
  auto pos1 = g.add_constant_pos_real(STD1);

  auto normal_dist1 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real1, pos1});

  const double MEAN2 = 0.0;
  const double STD2 = 2.0;
  auto real2 = g.add_constant_real(MEAN2);
  auto pos2 = g.add_constant_pos_real(STD2);

  auto normal_dist2 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real2, pos2});

  const double RATE1 = 3.0;
  auto rate1 = g.add_constant_pos_real(RATE1);

  auto poisson_dist1 = g.add_distribution(
      DistributionType::POISSON, AtomicType::NATURAL, std::vector<uint>{rate1});

  const double RATE2 = 5.0;
  auto rate2 = g.add_constant_pos_real(RATE2);

  auto poisson_dist2 = g.add_distribution(
      DistributionType::POISSON, AtomicType::NATURAL, std::vector<uint>{rate2});

  // Product parents must all have the same sample type
  // and agree with provided sample type.
  EXPECT_THROW(
      g.add_distribution(
          DistributionType::PRODUCT,
          AtomicType::NATURAL,
          std::vector<uint>{normal_dist1, poisson_dist1}),
      std::invalid_argument);

  EXPECT_THROW(
      g.add_distribution(
          DistributionType::PRODUCT,
          AtomicType::REAL, // ERRROR: does not agree with parents
          std::vector<uint>{poisson_dist1, poisson_dist2}),
      std::invalid_argument);

  auto product_dist1 = g.add_distribution(
      DistributionType::PRODUCT,
      AtomicType::REAL,
      std::vector<uint>{normal_dist1});

  auto product_dist2 = g.add_distribution(
      DistributionType::PRODUCT,
      AtomicType::UNKNOWN, // OK -- parents will determine
      std::vector<uint>{normal_dist1, normal_dist2});

  auto product_dist3 = g.add_distribution(
      DistributionType::PRODUCT,
      AtomicType::NATURAL,
      std::vector<uint>{poisson_dist1, poisson_dist2});

  auto dist_obj1 = static_cast<Distribution*>(g.node_ptrs()[product_dist1]);
  auto x = MEAN1;
  auto expected = log_normal_density(x, MEAN1, STD1);
  EXPECT_NEAR(dist_obj1->log_prob(NodeValue(x)), expected, 1e-6);

  auto dist_obj2 = static_cast<Distribution*>(g.node_ptrs()[product_dist2]);
  x = MEAN1;
  expected =
      log_normal_density(x, MEAN1, STD1) + log_normal_density(x, MEAN2, STD2);
  EXPECT_NEAR(dist_obj2->log_prob(NodeValue(x)), expected, 1e-6);

  auto dist_obj3 = static_cast<Distribution*>(g.node_ptrs()[product_dist3]);
  natural_t k = 4;
  expected =
      log_poisson_probability(k, RATE1) + log_poisson_probability(k, RATE2);
  EXPECT_NEAR(dist_obj3->log_prob(NodeValue(k)), expected, 1e-6);
}

void run_nmc_against_nuts_test(Graph& g) {
  auto max_abs_mean_diff = 0.1;
  // Increase number of rounds and observe
  // output to see the maximum absolute mean difference found
  // in case you need to adjust max_abs_mean_diff.
  auto num_rounds = 1;
  auto num_samples = 10000;
  auto warmup_samples = 1000;
  auto seed_getter = [] { return time(nullptr); };
  // Defining assert_near here because gtest macros
  // are not available in util.cpp where test_nmc_against_nuts is.
  auto assert_near = [&](double d1, double d2) {
    ASSERT_NEAR(d1, d2, max_abs_mean_diff);
  };

  test_nmc_against_nuts(
      g, num_rounds, num_samples, warmup_samples, seed_getter, assert_near);
}

TEST(testdistrib, product_inference_downstream) {
  // Defines a small model multiplying two normals,
  // and tests query on the sample of the product of
  // normals given upstream observations.
  Graph g;

  const double MEAN1 = -5.0;
  const double STD1 = 2.0;
  auto real1 = g.add_constant_real(MEAN1);
  auto pos1 = g.add_constant_pos_real(STD1);

  auto normal_dist1 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real1, pos1});

  const double MEAN2 = 5.0;
  const double STD2 = 2.0;
  auto real2 = g.add_constant_real(MEAN2);
  auto pos2 = g.add_constant_pos_real(STD2);

  auto normal_dist2 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real2, pos2});

  auto product_dist1 = g.add_distribution(
      DistributionType::PRODUCT,
      AtomicType::UNKNOWN,
      std::vector<uint>{normal_dist1, normal_dist2});

  auto product_sample =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{product_dist1});

  g.query(product_sample);

  run_nmc_against_nuts_test(g);
}

TEST(testdistrib, product_inference_upstream) {
  // Same as previous test but observing downstream
  // and querying upstream.
  Graph g;

  const double MEAN0 = -5.0;
  const double STD0 = 1.0;
  auto real0 = g.add_constant_real(MEAN0);
  auto pos0 = g.add_constant_pos_real(STD0);

  auto normal_dist0 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real0, pos0});

  auto real1 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{normal_dist0});

  const double STD1 = 2.0;
  auto pos1 = g.add_constant_pos_real(STD1);

  auto normal_dist1 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real1, pos1});

  const double MEAN2 = 5.0;
  const double STD2 = 2.0;
  auto real2 = g.add_constant_real(MEAN2);
  auto pos2 = g.add_constant_pos_real(STD2);

  auto normal_dist2 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real2, pos2});

  auto product_dist1 = g.add_distribution(
      DistributionType::PRODUCT,
      AtomicType::UNKNOWN,
      std::vector<uint>{normal_dist1, normal_dist2});

  auto product_sample1 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{product_dist1});

  auto product_sample2 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{product_dist1});

  auto product_sample3 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{product_dist1});

  g.observe(product_sample1, -1.0);
  g.observe(product_sample2, 0.0);
  g.observe(product_sample3, 1.0);

  g.query(real1);

  run_nmc_against_nuts_test(g);
}

TEST(testdistrib, product_inference_upstream_iid) {
  // Same test as product_inference_upstream,
  // but using a sample_idd on the product distribution instead
  // to test Product.*_iid methods.

  Graph g;

  const double MEAN0 = -5.0;
  const double STD0 = 1.0;
  auto real0 = g.add_constant_real(MEAN0);
  auto pos0 = g.add_constant_pos_real(STD0);

  auto normal_dist0 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real0, pos0});

  auto real1 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{normal_dist0});

  const double STD1 = 2.0;
  auto pos1 = g.add_constant_pos_real(STD1);

  auto normal_dist1 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real1, pos1});

  const double MEAN2 = 5.0;
  const double STD2 = 2.0;
  auto real2 = g.add_constant_real(MEAN2);
  auto pos2 = g.add_constant_pos_real(STD2);

  auto normal_dist2 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real2, pos2});

  auto product_dist1 = g.add_distribution(
      DistributionType::PRODUCT,
      AtomicType::UNKNOWN,
      std::vector<uint>{normal_dist1, normal_dist2});

  natural_t sample_size = 3;
  auto sample_size_node = g.add_constant_natural((natural_t)sample_size);
  auto product_sample_idd = g.add_operator(
      OperatorType::IID_SAMPLE,
      std::vector<uint>{product_dist1, sample_size_node});

  Eigen::MatrixXd data(sample_size, 1);
  data << -1.0, 0.0, 1.0;
  g.observe(product_sample_idd, data);

  g.query(real1);

  // TODO: NMC currently not working for this example
  // because it somehow selects
  // a _scalar_ proposer for the broadcast matrix
  // and an assertion is violated.
  // (Add task).
  // Running NUTS only and comparing with
  // result obtained from equivalent test
  // product_inference_upstream_iid
  // where iid samples are represented
  // as separate values.
  // run_nmc_against_nuts_test(g);

  auto expected = -2.848;
  NUTS nuts = NUTS(g);
  auto samples = nuts.infer(10000, time(nullptr), 1000);
  auto means_nuts = compute_means(samples);

  cout << means_nuts[0] << endl;

  ASSERT_NEAR(means_nuts[0], expected, 0.1);
}

TEST(testdistrib, product_inference_upstream_larger) {
  // A larger upstream inference test that builds a normal
  // from the product distribution and observes its sample.
  Graph g;

  const double MEAN0 = -5.0;
  const double STD0 = 1.0;
  auto real0 = g.add_constant_real(MEAN0);
  auto pos0 = g.add_constant_pos_real(STD0);

  auto normal_dist0 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real0, pos0});

  auto real1 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{normal_dist0});

  const double STD1 = 2.0;
  auto pos1 = g.add_constant_pos_real(STD1);

  auto normal_dist1 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real1, pos1});

  const double MEAN2 = 5.0;
  const double STD2 = 2.0;
  auto real2 = g.add_constant_real(MEAN2);
  auto pos2 = g.add_constant_pos_real(STD2);

  auto normal_dist2 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real2, pos2});

  auto product_dist1 = g.add_distribution(
      DistributionType::PRODUCT,
      AtomicType::UNKNOWN,
      std::vector<uint>{normal_dist1, normal_dist2});

  auto product_sample =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{product_dist1});

  const double STD3 = 3.0;
  auto two = g.add_constant_real(2.0);
  auto mean3 =
      g.add_operator(OperatorType::ADD, std::vector<uint>{product_sample, two});
  auto pos3 = g.add_constant_pos_real(STD3);

  auto normal_dist3 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{mean3, pos3});

  auto x1 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{normal_dist3});

  auto x2 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{normal_dist3});

  auto x3 =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{normal_dist3});

  g.observe(x1, 1.0);
  g.observe(x2, 2.0);
  g.observe(x3, 3.0);

  g.query(real1);

  run_nmc_against_nuts_test(g);
}
