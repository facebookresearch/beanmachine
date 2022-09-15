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
and checkes the results are close up to a maximum difference.
*/
void test_nmc_against_nuts(
    graph::Graph& graph,
    int num_rounds,
    int num_samples,
    int warmup_samples,
    unsigned seed,
    double max_abs_mean_diff);

} // namespace beanmachine::util
