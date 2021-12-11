/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <random>

#include "beanmachine/graph/distribution/tabular.h"

namespace beanmachine {
namespace distribution {

Tabular::Tabular(
    graph::AtomicType sample_type,
    const std::vector<graph::Node*>& in_nodes)
    : Distribution(graph::DistributionType::TABULAR, sample_type) {
  // check the sample datatype
  if (sample_type != graph::AtomicType::BOOLEAN) {
    throw std::invalid_argument("Tabular supports only boolean valued samples");
  }
  // extract the conditional probability vector from the first parent
  if (in_nodes.size() < 1 or
      in_nodes[0]->node_type != graph::NodeType::CONSTANT or
      in_nodes[0]->value.type.variable_type !=
          graph::VariableType::COL_SIMPLEX_MATRIX) {
    throw std::invalid_argument(
        "Tabular distribution's first arg must be COL_SIMPLEX_MATRIX");
  }
  const Eigen::MatrixXd& matrix = in_nodes[0]->value._matrix;
  // the matrix must have num rows = 2, since we only support BOOLEAN
  // sample_type
  if (matrix.rows() != 2) {
    throw std::invalid_argument(
        "Tabular distribution's first arg must have two rows.");
  }
  // the n_cols should be equal to 2^{num_parents}, since all parents are
  // boolean
  if (matrix.cols() != std::pow(2.0, (float)(in_nodes.size() - 1))) {
    throw std::invalid_argument(
        "Tabular distribution's first arg expected " +
        std::to_string((uint)std::pow(2.0, (float)(in_nodes.size() - 1))) +
        " dims got " + std::to_string(matrix.cols()));
  }
  // go through each of the parents other than the matrix and verify its type
  for (uint paridx = 1; paridx < static_cast<uint>(in_nodes.size()); paridx++) {
    const graph::Node* parent = in_nodes[paridx];
    if (parent->value.type != graph::AtomicType::BOOLEAN) {
      throw std::invalid_argument(
          "Tabular distribution only supports boolean parents currently");
    }
  }
}

double Tabular::get_probability() const {
  uint col_id = 0;
  uint row_id = 1;
  // map parents value to an index, starting from the last parent
  for (uint i = static_cast<uint>(in_nodes.size()) - 1, j = 0; i > 0;
       i--, j++) {
    const auto& parenti = in_nodes[i]->value;
    if (parenti.type != graph::AtomicType::BOOLEAN) {
      throw std::runtime_error(
          "Tabular distribution at node_id " + std::to_string(index) +
          " expects boolean parents");
    }
    if (parenti._bool) {
      col_id += (uint)std::pow(2.0, (float)j);
    }
  }
  assert(
      in_nodes[0]->value.type.variable_type ==
      graph::VariableType::COL_SIMPLEX_MATRIX);
  const Eigen::MatrixXd& matrix = in_nodes[0]->value._matrix;
  assert(col_id < matrix.cols());
  assert(row_id < matrix.rows());
  double prob = matrix.coeff(row_id, col_id);
  if (prob < 0 or prob > 1) {
    throw std::runtime_error(
        "unexpected probability " + std::to_string(prob) +
        " in Tabular node_id " + std::to_string(index));
  }
  return prob;
}

bool Tabular::_bool_sampler(std::mt19937& gen) const {
  double prob_true = get_probability();
  std::bernoulli_distribution dist(prob_true);
  return (bool)dist(gen);
}

double Tabular::log_prob(const graph::NodeValue& value) const {
  double prob_true = get_probability();
  if (value.type != graph::AtomicType::BOOLEAN) {
    throw std::runtime_error(
        "expecting boolean value in child of Tabular node_id " +
        std::to_string(index) + " got type " + value.type.to_string());
  }
  return value._bool ? std::log(prob_true) : std::log(1 - prob_true);
}

void Tabular::gradient_log_prob_value(
    const graph::NodeValue& /* value */,
    double& /* grad1 */,
    double& /* grad2 */) const {
  throw std::runtime_error(
      "gradient_log_prob_value not implemented for Tabular");
}

void Tabular::gradient_log_prob_param(
    const graph::NodeValue& /* value */,
    double& /* grad1 */,
    double& /* grad2 */) const {
  throw std::runtime_error(
      "gradient_log_prob_param not implemented for Tabular");
}

} // namespace distribution
} // namespace beanmachine
