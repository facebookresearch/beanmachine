/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <random>

#include "beanmachine/graph/proposer/gamma.h"

namespace beanmachine {
namespace proposer {

graph::NodeValue Gamma::sample(std::mt19937& gen) const {
  // C++ gamma has the second parameter as the scale rather than a rate
  std::gamma_distribution<double> dist(alpha, 1.0 / beta);
  return graph::NodeValue(graph::AtomicType::POS_REAL, dist(gen));
}

double Gamma::log_prob(graph::NodeValue& value) const {
  double x = value._double;
  return (alpha - 1) * std::log(x) - beta * x + alpha * std::log(beta) -
      std::lgamma(alpha);
}

} // namespace proposer
} // namespace beanmachine
