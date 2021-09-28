#include "beanmachine/graph/global/nuts.h"
#include "beanmachine/graph/global/proposer/nuts_proposer.h"
#include "beanmachine/graph/global/util.h"

namespace beanmachine {
namespace graph {

NUTS::NUTS(Graph& g) : GlobalMH(g), graph(g) {
  proposer = std::make_unique<NutsProposer>(NutsProposer());
}

void NUTS::prepare_graph() {
  set_default_transforms(graph);
}

} // namespace graph
} // namespace beanmachine
