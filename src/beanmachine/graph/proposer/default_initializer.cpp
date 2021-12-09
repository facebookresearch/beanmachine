/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "beanmachine/graph/proposer/default_initializer.h"
#include <stdexcept>
#include <string>
#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/operator/operator.h"
#include "beanmachine/graph/operator/stochasticop.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace proposer {

void default_initializer(std::mt19937& gen, graph::Node* node) {
  // The initialization rules here are based on Stan's default initialization
  // except for discrete variables which are sampled uniformly.
  // Note: Stan doesn't support discrete variables.
  if (node->value.type.variable_type ==
      graph::VariableType::COL_SIMPLEX_MATRIX) {
    // TODO: it seems like the initializer is needing to know
    // too much about the internal
    // structure of COL_SIMPLEX_MATRIX
    // (particularly, that it has an unconstrained_value).
    // It might make sense to create a derivation of NodeValue
    // for this type that keeps track of value and unconstrained value
    // internally and transparently.
    auto default_value = graph::NodeValue(node->value.type);
    node->value = default_value;
    auto sto_unobserved_node = static_cast<oper::StochasticOperator*>(node);
    sto_unobserved_node->unconstrained_value = default_value;
  } else if (node->value.type == graph::AtomicType::BOOLEAN) {
    bool val = std::bernoulli_distribution(0.5)(gen);
    node->value = graph::NodeValue(val);
  } else if (node->value.type == graph::AtomicType::PROBABILITY) {
    node->value = graph::NodeValue(graph::AtomicType::PROBABILITY, 0.5);
  } else if (node->value.type == graph::AtomicType::REAL) {
    node->value = graph::NodeValue(0.0);
  } else if (node->value.type == graph::AtomicType::POS_REAL) {
    node->value = graph::NodeValue(graph::AtomicType::POS_REAL, 1.0);
  } else {
    throw std::invalid_argument(
        "default initializer not defined for type " +
        node->value.type.to_string());
  }
}

} // namespace proposer
} // namespace beanmachine
