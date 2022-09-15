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

TEST(testglobal, global_hmc_stepsize_normal_normal) {
  int num_samples = 20000;
  bool adapt_mass_matrix = false;
  Graph g;
  auto expected_moments = build_normal_normal_model(g);
  HMC mh = HMC(g, 1.0, 0.5, adapt_mass_matrix);
  test_conjugate_model_moments(mh, expected_moments, num_samples);
}

TEST(testglobal, global_hmc_stepsize_gamma_gamma) {
  bool adapt_mass_matrix = false;
  Graph g;
  auto expected_moments = build_gamma_gamma_model(g);
  HMC mh = HMC(g, 1.0, 0.5, adapt_mass_matrix);
  test_conjugate_model_moments(mh, expected_moments);
}

TEST(testglobal, global_hmc_stepsize_gamma_normal) {
  bool adapt_mass_matrix = false;
  Graph g;
  auto expected_moments = build_gamma_normal_model(g);
  HMC mh = HMC(g, 1.0, 0.5, adapt_mass_matrix);
  test_conjugate_model_moments(mh, expected_moments);
}

TEST(testglobal, global_hmc_stepsize_beta_binomial) {
  // TODO: enable after supporting stickbreaking transform
  // int num_warmup_samples = 1000;
  // bool adapt_mass_matrix = false;
  // Graph g;
  // auto expected_moments = build_beta_binomial_model(g);
  // HMC mh = HMC(g, 1.0, 0.5, adapt_mass_matrix);
  // test_beta_binomial_model(mh, 5000, num_warmup_samples);
}

TEST(testglobal, global_hmc_stepsize_half_cauchy) {
  bool adapt_mass_matrix = false;
  Graph g;
  build_half_cauchy_model(g);
  HMC mh = HMC(g, 1.0, 0.5, adapt_mass_matrix);
  test_half_cauchy_model(mh);
}

TEST(testglobal, global_hmc_stepsize_mixed) {
  bool adapt_mass_matrix = false;
  Graph g;
  auto expected_moments = build_mixed_model(g);
  HMC mh = HMC(g, 0.5, 0.2, adapt_mass_matrix);
  test_conjugate_model_moments(mh, expected_moments);
}
