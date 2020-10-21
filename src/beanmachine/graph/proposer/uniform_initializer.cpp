// Copyright (c) Facebook, Inc. and its affiliates.
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace proposer {

graph::NodeValue uniform_initializer(std::mt19937& gen, graph::ValueType type) {
  // The initialization rules here are based on Stan's default initialization
  // except for discrete variables which are sampled uniformly.
  // Note: Stan doesn't support discrete variables.
  if (type == graph::AtomicType::BOOLEAN) {
    bool val = std::bernoulli_distribution(0.5)(gen);
    return graph::NodeValue(val);
  } else if (type == graph::AtomicType::PROBABILITY) {
    return graph::NodeValue(graph::AtomicType::PROBABILITY, 0.5);
  } else if (type == graph::AtomicType::REAL) {
    return graph::NodeValue(0.0);
  } else if (type == graph::AtomicType::POS_REAL) {
    return graph::NodeValue(graph::AtomicType::POS_REAL, 1.0);
  }
  // we shouldn't be called with other types, the following will invalidate the
  // value
  return graph::NodeValue();
}

} // namespace proposer
} // namespace beanmachine
