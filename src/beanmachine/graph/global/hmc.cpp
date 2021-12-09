/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/hmc.h"
#include "beanmachine/graph/global/proposer/hmc_proposer.h"
#include "beanmachine/graph/global/util.h"

namespace beanmachine {
namespace graph {

HMC::HMC(Graph& g, double path_length, double step_size)
    : GlobalMH(g), graph(g) {
  proposer = std::make_unique<HmcProposer>(HmcProposer(path_length, step_size));
}

void HMC::prepare_graph() {
  set_default_transforms(graph);
}

} // namespace graph
} // namespace beanmachine
