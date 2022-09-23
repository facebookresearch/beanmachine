/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

const double default_epsilon = 0.000001;

bool is_close(double a, double b, double eps = default_epsilon);

#define EXPECT_CLOSE(expected, actual)    \
  EXPECT_TRUE(is_close(expected, actual)) \
      << "expected " << expected << " actual " << actual
#define EXPECT_CLOSE_EPS(expected, actual, epsilon) \
  EXPECT_TRUE(is_close(expected, actual, epsilon))  \
      << "expected " << expected << " actual " << actual
