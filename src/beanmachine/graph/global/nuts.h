/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/global_mh.h"
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

/*
An implementation of the No-U-Turn Sampler as specified in [1]

Reference:
[1] Matthew Hoffman and Andrew Gelman. "The No-U-Turn Sampler: Adaptively
    Setting Path Lengths in Hamiltonian Monte Carlo" (2014).
    https://arxiv.org/abs/1111.4246
*/
class NUTS : public GlobalMH {
 public:
  explicit NUTS(
      Graph& g,
      bool adapt_mass_matrix = true,
      bool multinomial_sampling = true);
  /*
  NUTS by default transforms all unobserved random variables in the
  constrained space to the unconstrained space, similar to Stan in
  https://mc-stan.org/docs/2_27/reference-manual/variable-transforms-chapter.html

  Random variables of type POS_REAL have a LOG transform applied.
  */
  void prepare_graph() override;

 private:
  Graph& graph;
};

} // namespace graph
} // namespace beanmachine
