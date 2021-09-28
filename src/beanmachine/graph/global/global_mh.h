#include "beanmachine/graph/global/proposer/global_proposer.h"
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

class GlobalMH {
 public:
  Graph& graph;
  GlobalState state;
  std::unique_ptr<GlobalProposer> proposer;

  GlobalMH(Graph& g, uint seed);
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
  RandomWalkMH(Graph& g, uint seed, double step_size);
};

class HMC : public GlobalMH {
 public:
  HMC(Graph& g, uint seed, double path_length, double step_size);
};

class NUTS : public GlobalMH {
 public:
  NUTS(Graph& g, uint seed);
};

} // namespace graph
} // namespace beanmachine
