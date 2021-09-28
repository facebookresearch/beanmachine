#include "beanmachine/graph/global/global_mh.h"
#include "beanmachine/graph/global/proposer/nuts_proposer.h"

namespace beanmachine {
namespace graph {

/*
An implementation of the No-U-Turn Sampler as specified in [1]

Reference:
[1] Matthew Hoffman and Andrew Gelman. "The No-U-Turn Sampler: Adaptively
    Setting Path Lengths in Hamiltonian Monte Carlo" (2014).
    https://arxiv.org/abs/1111.4246
*/

NUTS::NUTS(Graph& g, uint seed) : GlobalMH(g, seed) {
  proposer = std::make_unique<NutsProposer>(NutsProposer());
}

} // namespace graph
} // namespace beanmachine
