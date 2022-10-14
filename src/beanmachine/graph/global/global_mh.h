/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <memory>
#include "beanmachine/graph/global/proposer/global_proposer.h"
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

class GlobalMH {
 public:
  explicit GlobalMH(std::unique_ptr<GlobalState> global_state);
  std::vector<std::vector<NodeValue>>& infer(
      int num_samples,
      uint seed,
      int num_warmup_samples = 0,
      bool save_warmup = false,
      InitType init_type = InitType::RANDOM);
  virtual void prepare_graph() {}
  void single_mh_step(GlobalState& state);
  virtual ~GlobalMH() {}

 private:
  std::unique_ptr<GlobalState> global_state;

 public:
  GlobalState& state;
  std::unique_ptr<GlobalProposer> proposer;
};

} // namespace graph
} // namespace beanmachine
