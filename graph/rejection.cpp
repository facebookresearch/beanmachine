// Copyright (c) Facebook, Inc. and its affiliates.
#include "beanmachine/graph/graph.h"

namespace beanmachine {
namespace graph {

void Graph::rejection(uint num_samples, std::mt19937& gen) {
  for (uint snum = 0; snum < num_samples; snum++) {
    // rejection sampling
    bool rejected;
    do {
      rejected = false;
      for (auto& node : nodes) {
        // We evaluate all the nodes in topological order so a node's
        // parents are evaluated before it.
        // Note: evaluation may result in sampling if there is a sample
        // operator in the graph.
        AtomicValue old_value;
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
    collect_sample();
  }
}

} // namespace graph
} // namespace beanmachine
