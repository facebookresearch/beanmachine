/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

const double eps = 0.00000001;

bool is_close(double a, double b);

#define EXPECT_CLOSE(expected, actual)    \
  EXPECT_TRUE(is_close(expected, actual)) \
      << "expected " << expected << " actual " << actual
