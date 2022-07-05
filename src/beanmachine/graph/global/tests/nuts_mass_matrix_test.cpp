/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "beanmachine/graph/global/nuts.h"
#include "beanmachine/graph/global/tests/conjugate_util_test.h"
#include "beanmachine/graph/graph.h"

using namespace beanmachine;
using namespace graph;

TEST(testglobal, global_nuts_mass_matrix_normal_normal) {
  bool adapt_mass_matrix = true;
  Graph g;
  auto expected_moments = build_normal_normal_model(g);
  NUTS mh = NUTS(g, adapt_mass_matrix);
  test_conjugate_model_moments(mh, expected_moments);
}

TEST(testglobal, global_nuts_mass_matrix_gamma_gamma) {
  bool adapt_mass_matrix = true;
  Graph g;
  auto expected_moments = build_gamma_gamma_model(g);
  NUTS mh = NUTS(g, adapt_mass_matrix);
  test_conjugate_model_moments(mh, expected_moments);
}

TEST(testglobal, global_nuts_mass_matrix_gamma_normal) {
  bool adapt_mass_matrix = true;
  Graph g;
  auto expected_moments = build_gamma_normal_model(g);
  NUTS mh = NUTS(g, adapt_mass_matrix);
  test_conjugate_model_moments(mh, expected_moments);
}

TEST(testglobal, global_nuts_mass_matrix_beta_binomial) {
  // TODO: enable after supporting stickbreaking transform
}

TEST(testglobal, global_nuts_mass_matrix_half_cauchy) {
  int num_warmup_samples = 2000;
  bool adapt_mass_matrix = true;
  double delta = 0.04;
  Graph g;
  build_half_cauchy_model(g);
  NUTS mh = NUTS(g, adapt_mass_matrix);
  test_half_cauchy_model(mh, num_warmup_samples, num_warmup_samples, delta);
}

TEST(testglobal, global_nuts_mass_matrix_mixed) {
  bool adapt_mass_matrix = true;
  Graph g;
  auto expected_moments = build_mixed_model(g);
  NUTS mh = NUTS(g, adapt_mass_matrix);
  test_conjugate_model_moments(mh, expected_moments);
}
