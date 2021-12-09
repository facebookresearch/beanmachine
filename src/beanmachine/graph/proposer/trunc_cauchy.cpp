/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <random>

#include "beanmachine/graph/proposer/trunc_cauchy.h"

namespace beanmachine {
namespace proposer {

TruncatedCauchy::TruncatedCauchy(double loc, double scale)
    : Proposer(), loc(loc), scale(scale) {
  // compute constant to be multiplied to the PDF to make the density integrate
  // to 1
  atan_0 = std::atan((0 - loc) / scale);
  // the inner term below is the integral of (1/s) * (1/(1+((x-loc)/scale)^2))
  // for x ranging from 0 to +infinity
  log_pdf_constant = -std::log(M_PI_2 - atan_0);
}

graph::NodeValue TruncatedCauchy::sample(std::mt19937& gen) const {
  std::uniform_real_distribution<double> dist(atan_0, M_PI_2);
  return graph::NodeValue(
      graph::AtomicType::POS_REAL, loc + scale * std::tan(dist(gen)));
}

double TruncatedCauchy::log_prob(graph::NodeValue& value) const {
  return log_pdf_constant - std::log(scale) -
      log(1 + std::pow((value._double - loc) / scale, 2));
}

} // namespace proposer
} // namespace beanmachine
