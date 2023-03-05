/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "beanmachine/graph/graph.h"

using namespace beanmachine;
using namespace graph;
TEST(testglobal, nuts_matrix_test) {
  // A very basic test which confirms that NUTS runs without crashing
  // on matrix samples. It makes no guarantees of sample validity.

  Graph g;
  auto one = g.add_constant_pos_real(1.0);
  auto lkj_chol_dist = g.add_distribution(
      DistributionType::LKJ_CHOLESKY,
      ValueType(VariableType::BROADCAST_MATRIX, AtomicType::REAL, 3, 3),
      std::vector<uint>{one});
  auto cov_llt =
      g.add_operator(OperatorType::SAMPLE, std::vector<uint>{lkj_chol_dist});
  g.query(cov_llt);
  auto samples = g.infer(2, InferenceType::NUTS);
  auto sample = samples[0][0];
  assert(sample._matrix.rows() == 3);
  assert(sample._matrix.cols() == 3);
}
