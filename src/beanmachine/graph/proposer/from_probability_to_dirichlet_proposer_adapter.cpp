/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>

#include "beanmachine/graph/graph.h"
#include "beanmachine/graph/proposer/from_probability_to_dirichlet_proposer_adapter.h"
#include "beanmachine/graph/proposer/proposer.h"

namespace beanmachine {
namespace proposer {

graph::NodeValue FromProbabilityToDirichletProposerAdapter::sample(
    std::mt19937& gen) const {
  auto probability = probability_proposer->sample(gen)._double;
  graph::ValueType value_type(
      graph::VariableType::COL_SIMPLEX_MATRIX,
      graph::AtomicType::PROBABILITY,
      2,
      1);
  Eigen::MatrixXd values(2, 1);
  values << probability, 1 - probability;
  graph::NodeValue dirichlet(value_type, values);
  return dirichlet;
}

double FromProbabilityToDirichletProposerAdapter::log_prob(
    graph::NodeValue& value) const {
  graph::NodeValue probability_node_value(
      graph::AtomicType::PROBABILITY, value._matrix.coeff(0));
  return probability_proposer->log_prob(probability_node_value);
}

} // namespace proposer
} // namespace beanmachine
