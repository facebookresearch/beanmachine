/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/proposer/global_proposer.h"

namespace beanmachine {
namespace graph {

class RandomWalkProposer : public GlobalProposer {
 public:
  explicit RandomWalkProposer(double step_size);
  double propose(GlobalState& state, std::mt19937& gen) override;

 private:
  double step_size;
};

} // namespace graph
} // namespace beanmachine
