#include "beanmachine/graph/global/global_state.h"
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

class GlobalProposer {
 public:
  explicit GlobalProposer() {}
  virtual double propose(GlobalState& state, std::mt19937& gen) = 0;
  virtual ~GlobalProposer() {}
};

} // namespace graph
} // namespace beanmachine
