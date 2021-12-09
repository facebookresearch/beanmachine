/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <random>

#include "beanmachine/graph/proposer/mixture.h"
#include "beanmachine/graph/proposer/normal.h"

using namespace beanmachine::graph;
using namespace beanmachine::proposer;

TEST(testproposer, mixture) {
  const double MU0 = 4.5;
  const double SIGMA0 = 3.4;
  const double WT0 = 10.1;
  const double MU1 = 13.5;
  const double SIGMA1 = 2.4;
  const double WT1 = 13.5;
  std::vector<double> weights;
  std::vector<std::unique_ptr<Proposer>> props;
  weights.push_back(WT0);
  props.push_back(std::make_unique<Normal>(MU0, SIGMA0));
  weights.push_back(WT1);
  props.push_back(std::make_unique<Normal>(MU1, SIGMA1));
  ASSERT_EQ(weights.size(), props.size());
  std::unique_ptr<Proposer> dist =
      std::make_unique<Mixture>(weights, std::move(props));
  std::mt19937 gen(31425);
  uint num_samples = 100000;
  double sum = 0.0;
  double sumsq = 0.0;
  for (uint i = 0; i < num_samples; i++) {
    auto val = dist->sample(gen);
    sum += val._double;
    sumsq += val._double * val._double;
  }
  double mean = sum / num_samples;
  double var = sumsq / num_samples - mean * mean;
  double exp_mean = (WT0 * MU0 + WT1 * MU1) / (WT0 + WT1);
  // see
  // https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
  double exp_var =
      (WT0 * SIGMA0 * SIGMA0 + WT1 * SIGMA1 * SIGMA1) / (WT0 + WT1) +
      (WT0 * MU0 * MU0 + WT1 * MU1 * MU1) / (WT0 + WT1) - exp_mean * exp_mean;
  EXPECT_NEAR(mean, exp_mean, 0.1);
  EXPECT_NEAR(var, exp_var, 0.1);
  NodeValue val(4.5);
  EXPECT_NEAR(dist->log_prob(val), -2.989, 0.01); // calculated in PyTorch
}
