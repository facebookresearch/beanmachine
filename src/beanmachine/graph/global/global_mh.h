#include "beanmachine/graph/global/proposer/global_proposer.h"
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

class GlobalMH {
 public:
  Graph& graph;
  GlobalState state;
  std::unique_ptr<GlobalProposer> proposer;

  explicit GlobalMH(Graph& g);
  std::vector<std::vector<NodeValue>>& infer(
      int num_samples,
      uint seed,
      int num_warmup_samples = 0,
      bool save_warmup = false);
  virtual void prepare_graph() {}
  void single_mh_step(GlobalState& state);
  virtual ~GlobalMH() {}
};

class RandomWalkMH : public GlobalMH {
 public:
  RandomWalkMH(Graph& g, double step_size);
};

class HMC : public GlobalMH {
 public:
  HMC(Graph& g, double path_length, double step_size);
};

class NUTS : public GlobalMH {
 public:
  explicit NUTS(Graph& g);
};

} // namespace graph
} // namespace beanmachine
