/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <stdexcept>

#include "beanmachine/graph/global/util.h"
#include "beanmachine/graph/graph.h"

using namespace beanmachine;
using namespace graph;

TEST(testglobal, global_default_transform) {
  Graph g;

  uint real_dist =
      g.add_distribution(DistributionType::FLAT, AtomicType::REAL, {});
  uint real_sample = g.add_operator(OperatorType::SAMPLE, {real_dist});
  g.query(real_sample);

  uint pos_real_dist =
      g.add_distribution(DistributionType::FLAT, AtomicType::POS_REAL, {});
  uint pos_real_sample = g.add_operator(OperatorType::SAMPLE, {pos_real_dist});
  g.query(pos_real_sample);

  set_default_transforms(g); // should run with no issues

  uint probability_dist =
      g.add_distribution(DistributionType::FLAT, AtomicType::PROBABILITY, {});
  uint probability_sample =
      g.add_operator(OperatorType::SAMPLE, {probability_dist});
  g.query(probability_sample);

  // TODO: add support for simplex distributions
  EXPECT_THROW(set_default_transforms(g), std::runtime_error);

  Graph g1;
  uint natural_dist =
      g1.add_distribution(DistributionType::FLAT, AtomicType::NATURAL, {});
  uint natural_sample = g1.add_operator(OperatorType::SAMPLE, {natural_dist});
  g1.query(natural_sample);
  EXPECT_THROW(set_default_transforms(g1), std::runtime_error);

  Graph g2;
  uint boolean_dist =
      g2.add_distribution(DistributionType::FLAT, AtomicType::BOOLEAN, {});
  uint boolean_sample = g2.add_operator(OperatorType::SAMPLE, {boolean_dist});
  g2.query(boolean_sample);
  EXPECT_THROW(set_default_transforms(g2), std::runtime_error);
}
