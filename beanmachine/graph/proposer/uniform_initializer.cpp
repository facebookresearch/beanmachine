// Copyright (c) Facebook, Inc. and its affiliates.
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace proposer {

graph::AtomicValue uniform_initializer(std::mt19937& gen, graph::AtomicType type) {
  if (type == graph::AtomicType::PROBABILITY) {
    double prob = util::sample_beta(gen, 1.0, 1.0);
    return graph::AtomicValue(graph::AtomicType::PROBABILITY, prob);
  }
  // we shouldn't be called with other types, the following will invalidate the value
  return graph::AtomicValue();
}

} // namespace proposer
} // namespace beanmachine
