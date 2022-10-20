/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/minibmg/inference/mle_inference.h"
#include <iostream>
#include <vector>
#include "beanmachine/minibmg/graph.h"
#include "beanmachine/minibmg/inference/hmc_world.h"

namespace beanmachine::minibmg {

std::vector<double> mle_inference_0(
    const Graph2& graph,
    double learning_rate,
    int num_rounds,
    std::vector<double> initial_proposals,
    bool print_progress) {
  auto abstraction = hmc_world_0(graph);
  int num_samples = abstraction->num_unobserved_samples();

  std::vector<double> proposals = initial_proposals;
  proposals.resize(num_samples);

  for (int round = 0; round < num_rounds; round++) {
    auto result = abstraction->evaluate(proposals);
    if (print_progress) {
      std::cout << fmt::format(
          "log_prob: {} inferred: {}\n",
          result.log_prob,
          abstraction->queries(proposals)[0]);
    }
    assert(!proposals.empty());
    for (int samp = 0; samp < num_samples; samp++) {
      // We use + rather than - here because we want to maximize (not minimize)
      // the log_prob; we move in the direction of the gradient rather than
      // opposite to it as we would in gradient descent.
      proposals[samp] += result.gradients[samp] * learning_rate;
    }
  }

  return abstraction->queries(proposals);
}

} // namespace beanmachine::minibmg
