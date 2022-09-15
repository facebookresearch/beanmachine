/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <memory>
#include "beanmachine/graph/global/nuts.h"
#include "beanmachine/graph/graph.h"
#include "graph/global/conjugate_util_test.h"

using namespace beanmachine;
using namespace graph;

TEST(testglobal, global_nuts_normal_normal) {
  int num_samples = 10000;
  int num_warmup_samples = 5000;
  bool adapt_mass_matrix = false;
  bool multinomial_sampling = false;
  Graph g;
  auto expected_moments = build_normal_normal_model(g);
  NUTS mh = NUTS(
      std::make_unique<GraphGlobalState>(g),
      adapt_mass_matrix,
      multinomial_sampling);
  test_conjugate_model_moments(
      mh, expected_moments, num_samples, num_warmup_samples);
}

TEST(testglobal, global_nuts_gamma_gamma) {
  int num_samples = 10000;
  int num_warmup = 5000;
  bool adapt_mass_matrix = false;
  bool multinomial_sampling = false;
  Graph g;
  auto expected_moments = build_gamma_gamma_model(g);
  NUTS mh = NUTS(
      std::make_unique<GraphGlobalState>(g),
      adapt_mass_matrix,
      multinomial_sampling);
  test_conjugate_model_moments(mh, expected_moments, num_samples, num_warmup);
}

TEST(testglobal, global_nuts_gamma_normal) {
  bool adapt_mass_matrix = false;
  bool multinomial_sampling = false;
  Graph g;
  auto expected_moments = build_gamma_normal_model(g);
  NUTS mh = NUTS(
      std::make_unique<GraphGlobalState>(g),
      adapt_mass_matrix,
      multinomial_sampling);
  test_conjugate_model_moments(mh, expected_moments);
}

TEST(testglobal, global_nuts_beta_binomial) {
  // TODO: enable after supporting stickbreaking transform
}

TEST(testglobal, global_nuts_half_cauchy) {
  bool adapt_mass_matrix = false;
  bool multinomial_sampling = false;
  Graph g;
  build_half_cauchy_model(g);
  NUTS mh = NUTS(
      std::make_unique<GraphGlobalState>(g),
      adapt_mass_matrix,
      multinomial_sampling);
  test_half_cauchy_model(mh);
}

TEST(testglobal, global_nuts_mixed) {
  bool adapt_mass_matrix = false;
  bool multinomial_sampling = false;
  Graph g;
  auto expected_moments = build_mixed_model(g);
  NUTS mh = NUTS(
      std::make_unique<GraphGlobalState>(g),
      adapt_mass_matrix,
      multinomial_sampling);
  test_conjugate_model_moments(mh, expected_moments);
}
