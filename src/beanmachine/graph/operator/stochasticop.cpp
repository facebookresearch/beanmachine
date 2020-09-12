// Copyright (c) Facebook, Inc. and its affiliates.

#include "beanmachine/graph/operator/stochasticop.h"

namespace beanmachine {
namespace oper {

Sample::Sample(const std::vector<graph::Node*>& in_nodes)
    : StochasticOperator(graph::OperatorType::SAMPLE) {
  if (in_nodes.size() != 1 or
      in_nodes[0]->node_type != graph::NodeType::DISTRIBUTION) {
    throw std::invalid_argument(
        "~ operator requires a single distribution parent");
  }
  const distribution::Distribution* dist =
      static_cast<distribution::Distribution*>(in_nodes[0]);
  // the type of value of a SAMPLE node is obviously the sample type
  // of the distribution parent
  value = graph::AtomicValue(dist->sample_type);
}

void Sample::eval(std::mt19937& gen) {
  const auto dist = static_cast<distribution::Distribution*>(in_nodes[0]);
  value = dist->sample(gen);
  return;
}

IIdSample::IIdSample(const std::vector<graph::Node*>& in_nodes)
    : StochasticOperator(graph::OperatorType::IID_SAMPLE) {
  if (in_nodes.size() != 2 ) {
    throw std::invalid_argument(
        "iid sample operator requires exactly 2 parent nodes");
  }
  if (in_nodes[0]->node_type != graph::NodeType::DISTRIBUTION) {
    throw std::invalid_argument(
      "for iid sample, the 1st parent must be a distribution node");
  }
  if (in_nodes[1]->node_type != graph::NodeType::CONSTANT or
      in_nodes[1]->value.type != graph::AtomicType::NATURAL) {
    throw std::invalid_argument(
      "for iid sample, the 2nd parent must be a constant natural-valued node");
  }
  if (in_nodes[1]->value._natural < 2) {
    throw std::invalid_argument(
      "for iid sample, the 2nd parent must have value >= 2");
  }
  const auto dist = static_cast<distribution::Distribution*>(in_nodes[0]);
  if (dist->dist_type != graph::DistributionType::BETA) {
    throw std::invalid_argument(
      "currently iid sample only supports Beta distribution");
  }
  // determine the value type
  graph::ValueType vtype;
  switch (dist->sample_type.variable_type) {
    case graph::VariableType::SCALAR:
      vtype = graph::ValueType(
          graph::VariableType::BROADCAST_MATRIX,
          dist->sample_type.atomic_type,
          in_nodes[1]->value._natural,
          1);
      break;
    case graph::VariableType::BROADCAST_MATRIX:
    case graph::VariableType::COL_SIMPLEX_MATRIX:
      //TODO(ddeng): add IID_SAMPLE_COL after refactoring Operator class
      throw std::invalid_argument(
          "For matrix sample types, use IID_SAMPLE_COL. ");
    default:
      throw std::invalid_argument(
          "Invalid sample type for for iid sample. ");
  }
  value = graph::AtomicValue(vtype);
  return;
}

void IIdSample::eval(std::mt19937& gen) {
  const auto dist = static_cast<distribution::Distribution*>(in_nodes[0]);
  // TODO(ddeng): add sample(gen, value) to all dist tyes.
  dist->sample(gen, value);
  return;
}

} // namespace oper
} // namespace beanmachine
