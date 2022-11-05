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
    const Graph& graph,
    double learning_rate,
    int num_rounds,
    std::vector<double> initial_proposals,
    bool print_progress) {
  auto abstraction = hmc_world_0(graph);
  int num_samples = abstraction->num_unobserved_samples();

  std::vector<double> proposals = initial_proposals;
  proposals.resize(num_samples);

  for (int round = 0; round < num_rounds; round++) {
    std::vector<double> grads;
    abstraction->gradients(proposals, grads);
    if (print_progress) {
      std::vector<double> queries;
      abstraction->queries(proposals, queries);
      assert(!queries.empty());
      std::cout << fmt::format(
          "log_prob: {} inferred: {}\n",
          abstraction->log_prob(proposals),
          queries[0]);
    }
    assert(!proposals.empty());
    for (int samp = 0; samp < num_samples; samp++) {
      // We use + rather than - here because we want to maximize (not minimize)
      // the log_prob; we move in the direction of the gradient rather than
      // opposite to it as we would in gradient descent.
      proposals[samp] += grads[samp] * learning_rate;
    }
  }

  std::vector<double> queries;
  abstraction->queries(proposals, queries);
  return queries;
}

} // namespace beanmachine::minibmg
