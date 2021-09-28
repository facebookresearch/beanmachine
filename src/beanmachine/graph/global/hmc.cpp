#include "beanmachine/graph/global/global_mh.h"
#include "beanmachine/graph/global/proposer/hmc_proposer.h"

namespace beanmachine {
namespace graph {

HMC::HMC(Graph& g, uint seed, double path_length, double step_size)
    : GlobalMH(g, seed) {
  proposer = std::make_unique<HmcProposer>(HmcProposer(path_length, step_size));
}

} // namespace graph
} // namespace beanmachine
