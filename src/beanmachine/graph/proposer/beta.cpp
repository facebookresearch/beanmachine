/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include "beanmachine/graph/proposer/beta.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace proposer {

graph::NodeValue Beta::sample(std::mt19937& gen) const {
  return graph::NodeValue(
      graph::AtomicType::PROBABILITY, util::sample_beta(gen, a, b));
}

double Beta::log_prob(graph::NodeValue& value) const {
  double ret_val =
      (a - 1) * log(value._double) + (b - 1) * log(1 - value._double);
  ret_val += lgamma(a + b) - lgamma(a) - lgamma(b);
  return ret_val;
}

} // namespace proposer
} // namespace beanmachine
