/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "beanmachine/graph/graph.h"

#include <functional>

namespace beanmachine::util {

/*
Returns the mean of the index-th dimension in samples.
*/
double compute_mean_at_index(
    std::vector<std::vector<graph::NodeValue>> samples,
    std::size_t index);

/*
Returns the means of samples.
*/
std::vector<double> compute_means(
    std::vector<std::vector<graph::NodeValue>> samples);

/*
Runs both NMC and NUTS on given graph for a number of rounds
with a given number of samples (warmup samples is used only for NUTS),
and calls a tester function on the means of the first query
variable obtained by both algorithms.
The seed is provided as a nullary function (so it can vary across rounds).
Also prints the obtained means and the measured maximum difference over all
rounds.
*/
void test_nmc_against_nuts(
    graph::Graph& graph,
    int num_rounds,
    int num_samples,
    int warmup_samples,
    std::function<unsigned()> seed_getter,
    std::function<void(double, double)> tester);

} // namespace beanmachine::util
