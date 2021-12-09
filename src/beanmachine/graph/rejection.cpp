/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

void Graph::rejection(uint num_samples, uint seed, InferConfig infer_config) {
  std::mt19937 gen(seed);
  std::vector<Node*> ordered_supp;
  if (infer_config.keep_log_prob) {
    std::set<uint> supp = compute_support();
    for (uint node_id : supp) {
      ordered_supp.push_back(nodes[static_cast<uint>(node_id)].get());
    }
  }
  for (uint snum = 0; snum < num_samples + infer_config.num_warmup; snum++) {
    // rejection sampling
    bool rejected;
    do {
      rejected = false;
      for (auto& node : nodes) {
        // We evaluate all the nodes in topological order so a node's
        // parents are evaluated before it.
        // Note: evaluation may result in sampling if there is a sample
        // operator in the graph.
        NodeValue old_value;
        if (node->node_type == NodeType::OPERATOR) {
          old_value = node->value;
          node->eval(gen);
        }
        if (observed.find(node->index) != observed.end()) {
          // we can't change the value of the observed nodes
          // sample is rejected if observed value doesn't match up
          if (old_value != node->value) {
            node->value = old_value;
            rejected = true;
            break;
          }
        }
      }
    } while (rejected);
    if (infer_config.keep_log_prob) {
      collect_log_prob(_full_log_prob(ordered_supp));
    }
    if (infer_config.keep_warmup or snum >= infer_config.num_warmup) {
      collect_sample();
    }
  }
}

} // namespace graph
} // namespace beanmachine
