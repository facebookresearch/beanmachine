/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>
#include "beanmachine/minibmg/graph2.h"

namespace beanmachine::minibmg {

/*
 * A hacked-together and totally inadequate inference method that computes a
 * local maxima of the log probability, mainly provided for testing `hmc_world`.
 * I hope that's what "MLE" inference means (Maximum Likelihood Estimation)
 * otherwise I'll be totally embarrased.  This implementation is not adaptive;
 * it requires you to set the learning rate and number of rounds.  In practice
 * appropriate values for these vary wildly from application to application.
 * Perhaps someday we will do a better job of inferring them by computing the
 * second derivative, utilizing momentum, checking for convergence, and/or
 * something else.
 *
 * This is version zero (0) because it exercises `hmc_world_0()`.  In the future
 * we'll exercise more advanced versions of that method.
 */
std::vector<double> mle_inference_0(
    const Graph2& graph,
    double learning_rate = 0.1,
    int num_rounds = 25,
    std::vector<double> initial_proposals = std::vector<double>{},
    bool print_progress = false);

} // namespace beanmachine::minibmg
