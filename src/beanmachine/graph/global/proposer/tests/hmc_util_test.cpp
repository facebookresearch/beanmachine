/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <random>

#include "beanmachine/graph/global/proposer/hmc_util.h"

using namespace beanmachine;
using namespace graph;

TEST(testglobal, hmc_util_step_size) {
  StepSizeAdapter step_size_adapter = StepSizeAdapter(0.65);
  step_size_adapter.initialize(1.0);
  step_size_adapter.update_step_size(0.5);
  EXPECT_NEAR(step_size_adapter.finalize_step_size(), 7.613, 1e-4);

  step_size_adapter.initialize(0.5);
  step_size_adapter.update_step_size(0.0);
  step_size_adapter.update_step_size(0.9);
  EXPECT_NEAR(step_size_adapter.finalize_step_size(), 1.7678, 1e-4);
}
