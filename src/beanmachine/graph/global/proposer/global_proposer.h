/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "beanmachine/graph/global/global_state.h"

namespace beanmachine {
namespace graph {

class GlobalProposer {
 public:
  explicit GlobalProposer() {}
  virtual void warmup(
      GlobalState& /*state*/,
      std::mt19937& /*gen*/,
      double /*acceptance_log_prob*/,
      int /*iteration*/,
      int /*num_warmup_samples*/) {}
  virtual double propose(GlobalState& state, std::mt19937& gen) = 0;
  virtual void initialize(
      GlobalState& /*state*/,
      std::mt19937& /*gen*/,
      int /*num_warmup_samples*/) {}
  virtual ~GlobalProposer() {}
};

} // namespace graph
} // namespace beanmachine
