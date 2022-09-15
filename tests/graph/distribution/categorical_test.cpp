/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "beanmachine/graph/distribution/categorical.h"
#include "beanmachine/graph/graph.h"

using namespace beanmachine;

TEST(testdistrib, categorical) {
  graph::Graph g;

  // 50% chance of 0, 25% chance of 1, 12.5% chance of 3, 4.
  Eigen::MatrixXd matrix(4, 1);
  matrix << 0.5, 0.25, 0.125, 0.125;
  // We only support single-column simplexes.
  Eigen::MatrixXd matrix2(4, 2);
  matrix2 << 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25;

  uint c0 = g.add_constant(0.0);
  uint c1 = g.add_constant_col_simplex_matrix(matrix);
  uint c2 = g.add_constant_col_simplex_matrix(matrix2);

  // Negative test: bad return type
  EXPECT_THROW(
      g.add_distribution(
          graph::DistributionType::CATEGORICAL,
          graph::AtomicType::BOOLEAN,
          std::vector<uint>({c1})),
      std::invalid_argument);

  // Negative test: wrong argument counts:
  EXPECT_THROW(
      g.add_distribution(
          graph::DistributionType::CATEGORICAL,
          graph::AtomicType::NATURAL,
          std::vector<uint>({})),
      std::invalid_argument);

  EXPECT_THROW(
      g.add_distribution(
          graph::DistributionType::CATEGORICAL,
          graph::AtomicType::NATURAL,
          std::vector<uint>({c1, c1})),
      std::invalid_argument);

  // Negative test: wrong argument type
  EXPECT_THROW(
      g.add_distribution(
          graph::DistributionType::CATEGORICAL,
          graph::AtomicType::NATURAL,
          std::vector<uint>({c0})),
      std::invalid_argument);

  // Negative test: wrong simplex dimensionality
  EXPECT_THROW(
      g.add_distribution(
          graph::DistributionType::CATEGORICAL,
          graph::AtomicType::NATURAL,
          std::vector<uint>({c2})),
      std::invalid_argument);

  // Positive test: construct a categorical correctly:
  uint d1 = g.add_distribution(
      graph::DistributionType::CATEGORICAL,
      graph::AtomicType::NATURAL,
      std::vector<uint>({c1}));

  uint s1 = g.add_operator(graph::OperatorType::SAMPLE, std::vector<uint>{d1});

  // There should be a 0.125 chance of observing a 3:
  g.observe(s1, (graph::natural_t)3);
  EXPECT_NEAR(g.full_log_prob(), log(0.125), 1e-3);

  // TODO: test IID_SAMPLE
  // TODO: test g.eval_and_grad
}
