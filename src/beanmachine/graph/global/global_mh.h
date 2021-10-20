#pragma once
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
      unsigned int seed,
      int num_warmup_samples = 0,
      bool save_warmup = false,
      InitType init_type = InitType::RANDOM);
  virtual void prepare_graph() {}
  void single_mh_step(GlobalState& state);
  virtual ~GlobalMH() {}
};

} // namespace graph
} // namespace beanmachine
