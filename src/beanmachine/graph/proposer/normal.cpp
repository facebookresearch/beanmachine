/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define _USE_MATH_DEFINES
#include <cmath>

#include <random>

#include "beanmachine/graph/proposer/normal.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace proposer {

graph::NodeValue Normal::sample(std::mt19937& gen) const {
  std::normal_distribution<double> dist(mu, sigma);
  return graph::NodeValue(graph::AtomicType::REAL, dist(gen));
}

double Normal::log_prob(graph::NodeValue& value) const {
  return util::log_normal_density(value._double, mu, sigma);
}

} // namespace proposer
} // namespace beanmachine
