/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/nuts.h"
#include <memory>
#include "beanmachine/graph/global/proposer/nuts_proposer.h"
#include "beanmachine/graph/global/util.h"

namespace beanmachine {
namespace graph {

NUTS::NUTS(
    std::unique_ptr<GlobalState> state,
    bool adapt_mass_matrix,
    bool multinomial_sampling)
    : GlobalMH(std::move(state)) {
  proposer =
      std::make_unique<NutsProposer>(adapt_mass_matrix, multinomial_sampling);
}
NUTS::NUTS(Graph& graph, bool adapt_mass_matrix, bool multinomial_sampling)
    : GlobalMH(std::make_unique<GraphGlobalState>(graph)) {
  proposer =
      std::make_unique<NutsProposer>(adapt_mass_matrix, multinomial_sampling);
}

void NUTS::prepare_graph() {
  state.set_default_transforms();
}

void Graph::nuts(uint num_samples, uint seed, InferConfig infer_config) {
  NUTS(std::make_unique<GraphGlobalState>(*this))
      .infer(
          num_samples, seed, infer_config.num_warmup, infer_config.keep_warmup);
}

} // namespace graph
} // namespace beanmachine
