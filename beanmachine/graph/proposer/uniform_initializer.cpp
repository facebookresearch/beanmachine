// Copyright (c) Facebook, Inc. and its affiliates.
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace proposer {

graph::AtomicValue uniform_initializer(
    std::mt19937& gen,
    graph::AtomicType type) {
  // The initialization rules here are based on Stan's default initialization
  // except for discrete variables which are sampled uniformly.
  // Note: Stan doesn't support discrete variables.
  if (type == graph::AtomicType::BOOLEAN) {
    bool val = std::bernoulli_distribution(0.5)(gen);
    return graph::AtomicValue(val);
  } else if (type == graph::AtomicType::PROBABILITY) {
    return graph::AtomicValue(graph::AtomicType::PROBABILITY, 0.5);
  } else if (type == graph::AtomicType::REAL) {
    return graph::AtomicValue(0.0);
  } else if (type == graph::AtomicType::POS_REAL) {
    return graph::AtomicValue(graph::AtomicType::POS_REAL, 1.0);
  }
  // we shouldn't be called with other types, the following will invalidate the
  // value
  return graph::AtomicValue();
}

} // namespace proposer
} // namespace beanmachine
