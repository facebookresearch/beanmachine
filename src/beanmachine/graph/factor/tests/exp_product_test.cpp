/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"

using namespace beanmachine::graph;

TEST(testfactor, exp_product) {
  Graph g;
  // negative test, exp_product has at least one parent
  EXPECT_THROW(
      g.add_factor(FactorType::EXP_PRODUCT, std::vector<uint>{}),
      std::invalid_argument);
  // negative test, booleans can't be the parent of an exp_product
  uint bool1 = g.add_constant(true);
  EXPECT_THROW(
      g.add_factor(FactorType::EXP_PRODUCT, std::vector<uint>{bool1}),
      std::invalid_argument);
  // positive test, mixed product types are allowed
  uint pos1 = g.add_constant_pos_real(2.0);
  uint real1 = g.add_constant(3.0);
  uint prob1 = g.add_constant_probability(0.4);
  uint neg1 = g.add_constant_neg_real(-1.0);
  g.add_factor(
      FactorType::EXP_PRODUCT,
      std::vector<uint>{pos1, real1, prob1, neg1, neg1});
  uint dist1 = g.add_distribution(
      DistributionType::NORMAL,
      AtomicType::REAL,
      std::vector<uint>{real1, pos1});
  uint x = g.add_operator(OperatorType::SAMPLE, std::vector<uint>{dist1});
  uint x_sq = g.add_operator(OperatorType::MULTIPLY, std::vector<uint>{x, x});
  g.add_factor(FactorType::EXP_PRODUCT, std::vector<uint>{x, prob1, x_sq});
  g.observe(x, 7.0);
  // f(x) = -0.5 (x-3)^2 / 2^2 + 0.4 x^3
  // f'(x) = -(x-3)/4 + 1.2 x^2
  // f'(7) = 57.8
  // f''(x) = -1/4 + 2.4 x
  // f''(7) = 16.55
  double grad1 = 0;
  double grad2 = 0;
  g.gradient_log_prob(x, grad1, grad2);
  EXPECT_NEAR(grad1, 57.8, 1e-3);
  EXPECT_NEAR(grad2, 16.55, 1e-3);
  std::vector<DoubleMatrix*> grad;
  g.eval_and_grad(grad);
  EXPECT_EQ(grad.size(), 1);
  EXPECT_NEAR((*grad[0]), 57.8, 1e-3);
}
