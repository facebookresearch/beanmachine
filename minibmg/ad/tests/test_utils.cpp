/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/ad/tests/test_utils.h"
#include <cmath>

using namespace std;

bool is_close(double a, double b, double eps) {
  if (isnan(a) != isnan(b))
    return false;
  if (isnan(a) && isnan(b))
    return true;
  return a == b || std::abs(a - b) < eps * (std::abs(a) + std::abs(b));
}
