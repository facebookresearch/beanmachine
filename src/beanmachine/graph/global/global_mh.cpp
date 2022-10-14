/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/global/global_mh.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace graph {

GlobalMH::GlobalMH(std::unique_ptr<GlobalState> global_state0)
    : global_state(std::move(global_state0)), state{*global_state} {}

std::vector<std::vector<NodeValue>>& GlobalMH::infer(
    int num_samples,
    uint seed,
    int num_warmup_samples,
    bool save_warmup,
    InitType init_type) {
  std::mt19937 gen(seed);
  // TODO: tie samples directly to inference
  state.set_agg_type(AggregationType::NONE);
  state.clear_samples();

  prepare_graph();
  state.initialize_values(init_type, seed);
  proposer->initialize(state, gen, num_warmup_samples);

  for (int i = 0; i < num_samples + num_warmup_samples; i++) {
    double acceptance_log_prob = proposer->propose(state, gen);
    bool accept_sample =
        util::flip_coin_with_log_prob(gen, acceptance_log_prob);

    if (accept_sample) {
      // backup new samples + grads for future proposals to revert to
      // Note: we are backing up only when the samples have changed
      // for the sake of performance
      state.backup_unconstrained_values();
      state.backup_unconstrained_grads();
    } else {
      // revert to previously backed up samples + grads
      state.revert_unconstrained_values();
      state.revert_unconstrained_grads();
      state.update_log_prob();
    }

    if (i < num_warmup_samples) {
      double acceptance_prob = std::min(std::exp(acceptance_log_prob), 1.0);
      proposer->warmup(state, gen, acceptance_prob, i + 1, num_warmup_samples);
      if (save_warmup) {
        state.collect_sample();
      }
    } else {
      state.collect_sample();
    }
  }

  return state.get_samples();
}

} // namespace graph
} // namespace beanmachine
