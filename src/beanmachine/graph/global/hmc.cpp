/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/hmc.h"
#include <memory>
#include "beanmachine/graph/global/proposer/hmc_proposer.h"
#include "beanmachine/graph/global/util.h"

namespace beanmachine {
namespace graph {

HMC::HMC(
    std::unique_ptr<GlobalState> global_state,
    double path_length,
    double step_size,
    bool adapt_mass_matrix)
    : GlobalMH(std::move(global_state)) {
  proposer =
      std::make_unique<HmcProposer>(path_length, step_size, adapt_mass_matrix);
}
HMC::HMC(
    Graph& graph,
    double path_length,
    double step_size,
    bool adapt_mass_matrix)
    : GlobalMH(std::make_unique<GraphGlobalState>(graph)) {
  proposer =
      std::make_unique<HmcProposer>(path_length, step_size, adapt_mass_matrix);
}

void HMC::prepare_graph() {
  state.set_default_transforms();
}

} // namespace graph
} // namespace beanmachine
