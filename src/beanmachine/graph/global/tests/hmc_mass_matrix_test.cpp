/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

TEST(testglobal, global_hmc_mass_matrix_normal_normal) {
  int num_samples = 10000;
  int num_warmup_samples = 1000;
  bool adapt_mass_matrix = true;
  Graph g;
  auto expected_moments = build_normal_normal_model(g);
  HMC mh = HMC(g, 1.0, 0.5, adapt_mass_matrix);
  test_conjugate_model_moments(
      mh, expected_moments, num_samples, num_warmup_samples);
}

TEST(testglobal, global_hmc_mass_matrix_gamma_gamma) {
  int num_samples = 10000;
  int num_warmup_samples = 5000;
  bool adapt_mass_matrix = true;
  Graph g;
  auto expected_moments = build_gamma_gamma_model(g);
  HMC mh = HMC(g, 1.0, 0.5, adapt_mass_matrix);
  test_conjugate_model_moments(
      mh, expected_moments, num_samples, num_warmup_samples);
}

TEST(testglobal, global_hmc_mass_matrix_beta_binomial) {
  // TODO: enable after supporting stickbreaking transform
}

TEST(testglobal, global_hmc_mass_matrix_half_cauchy) {
  int num_samples = 5000;
  int num_warmup_samples = 1000;
  bool adapt_mass_matrix = true;
  Graph g;
  build_half_cauchy_model(g);
  HMC mh = HMC(g, 1.0, 0.5, adapt_mass_matrix);
  // TODO: fix this in next diff
  test_half_cauchy_model(mh, num_samples, num_warmup_samples, 1.0);
}

TEST(testglobal, global_hmc_mass_matrix_mixed) {
  bool adapt_mass_matrix = true;
  Graph g;
  auto expected_moments = build_mixed_model(g);
  HMC mh = HMC(g, 0.1, 0.05, adapt_mass_matrix);
  test_conjugate_model_moments(mh, expected_moments);
}
