#include "beanmachine/graph/global/global_mh.h"
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

class HMC : public GlobalMH {
 public:
  HMC(Graph& g, double path_length, double step_size);
  /*
  HMC by default transforms all unobserved random variables in the
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
