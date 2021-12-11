/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <random>

#include "beanmachine/graph/proposer/mixture.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace proposer {

graph::NodeValue Mixture::sample(std::mt19937& gen) const {
  // we need to pick one of the proposers proportional to their weight
  // so we sample a target value between 0 and sum(i=1..n, w_i)
  // then we pick the index j s.t. sum(i=1..j-1, w_i) <= target < sum(i=1..j,
  // w_i)
  std::uniform_real_distribution<double> dist(0, 1);
  double target = weight_sum * dist(gen);
  std::vector<double>::size_type index = 0;
  double sum = 0;
  for (; index < weights.size(); index++) {
    sum += weights[index];
    if (target < sum) {
      break;
    }
  }
  // due to numerical stability issues we could have gone past all the
  // elements in this case pick the last element
  if (index == weights.size()) {
    index--;
  }
  return proposers[index]->sample(gen);
}

double Mixture::log_prob(graph::NodeValue& value) const {
  std::vector<double> log_probs;
  for (std::vector<double>::size_type index = 0; index < weights.size();
       index++) {
    log_probs.push_back(
        log(weights[index]) + proposers[index]->log_prob(value));
  }
  return util::log_sum_exp(log_probs) - std::log(weight_sum);
}

} // namespace proposer
} // namespace beanmachine
