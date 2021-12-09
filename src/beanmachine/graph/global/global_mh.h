/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
      uint seed,
      int num_warmup_samples = 0,
      bool save_warmup = false,
      InitType init_type = InitType::RANDOM);
  virtual void prepare_graph() {}
  void single_mh_step(GlobalState& state);
  virtual ~GlobalMH() {}
};

} // namespace graph
} // namespace beanmachine
