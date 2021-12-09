/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <string>

#include "beanmachine/graph/distribution/flat.h"

namespace beanmachine {
namespace distribution {

using namespace graph;

Flat::Flat(AtomicType sample_type, const std::vector<Node*>& in_nodes)
    : Distribution(DistributionType::FLAT, sample_type) {
  // a Flat distribution has no parents
  if (in_nodes.size() != 0) {
    throw std::invalid_argument("Flat distribution has no parents");
  }
}

Flat::Flat(ValueType sample_type, const std::vector<Node*>& in_nodes)
    : Distribution(DistributionType::FLAT, sample_type) {
  // a Flat distribution has no parents
  if (in_nodes.size() != 0) {
    throw std::invalid_argument("Flat distribution has no parents");
  }
}

bool Flat::_bool_sampler(std::mt19937& gen) const {
  std::bernoulli_distribution dist(0.5);
  return (bool)dist(gen);
}

std::uniform_real_distribution<double> Flat::_get_uniform_real_distribution()
    const {
  std::uniform_real_distribution<double> dist;
  switch (sample_type.atomic_type) {
    case graph::AtomicType::REAL:
      dist = std::uniform_real_distribution<double>(
          std::numeric_limits<double>::lowest(),
          std::numeric_limits<double>::max());
      break;
    case graph::AtomicType::POS_REAL:
      dist = std::uniform_real_distribution<double>(
          0, std::numeric_limits<double>::max());
      break;
    case graph::AtomicType::PROBABILITY:
      dist = std::uniform_real_distribution<double>(0, 1);
      break;
    default:
      throw std::runtime_error(
          "Unsupported sample type for _double_sampler of Flat.");
  }
  return dist;
}

double Flat::_double_sampler(std::mt19937& gen) const {
  return _get_uniform_real_distribution()(gen);
}

natural_t Flat::_natural_sampler(std::mt19937& gen) const {
  std::uniform_int_distribution<natural_t> dist(
      0, std::numeric_limits<natural_t>::max());
  return (natural_t)dist(gen);
}

Eigen::MatrixXd Flat::_matrix_sampler(std::mt19937& gen) const {
  int rows = static_cast<int>(sample_type.rows);
  int cols = static_cast<int>(sample_type.cols);
  Eigen::MatrixXd result(rows, cols);
  std::uniform_real_distribution<double> dist =
      _get_uniform_real_distribution();
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      result(i, j) = dist(gen);
    }
  }
  return result;
}

// A Flat distribution is really easy in terms of computing the log_prob and the
// gradients of the log_prob. These are all zero!

double Flat::log_prob(const NodeValue& /* value */) const {
  return 0;
}

void Flat::gradient_log_prob_value(
    const NodeValue& /* value */,
    double& /* grad1 */,
    double& /* grad2 */) const {}

void Flat::gradient_log_prob_param(
    const NodeValue& /* value */,
    double& /* grad1 */,
    double& /* grad2 */) const {}

} // namespace distribution
} // namespace beanmachine
