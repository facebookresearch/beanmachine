#include "beanmachine/graph/global/global_mh.h"
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

class RandomWalkMH : public GlobalMH {
 public:
  RandomWalkMH(Graph& g, double step_size);
};

} // namespace graph
} // namespace beanmachine
