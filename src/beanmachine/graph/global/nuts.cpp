/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/nuts.h"
#include "beanmachine/graph/global/proposer/nuts_proposer.h"
#include "beanmachine/graph/global/util.h"

namespace beanmachine {
namespace graph {

NUTS::NUTS(Graph& g, bool adapt_mass_matrix, bool multinomial_sampling)
    : GlobalMH(g), graph(g) {
  proposer = std::make_unique<NutsProposer>(
      NutsProposer(adapt_mass_matrix, multinomial_sampling));
}

void NUTS::prepare_graph() {
  set_default_transforms(graph);
}

void Graph::nuts(uint num_samples, uint seed, InferConfig infer_config) {
  NUTS(*this).infer(
      num_samples, seed, infer_config.num_warmup, infer_config.keep_warmup);
}

} // namespace graph
} // namespace beanmachine
