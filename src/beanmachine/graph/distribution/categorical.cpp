/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include "beanmachine/graph/distribution/categorical.h"

namespace beanmachine {
namespace distribution {

Categorical::Categorical(
    graph::AtomicType sample_type,
    const std::vector<graph::Node*>& in_nodes)
    : Distribution(graph::DistributionType::CATEGORICAL, sample_type) {
  if (sample_type != graph::AtomicType::NATURAL) {
    throw std::invalid_argument("Categorical produces natural valued samples");
  }
  if (in_nodes.size() != 1) {
    throw std::invalid_argument(
        "Categorical distribution must have exactly one parent");
  }
  const auto& parent0 = in_nodes[0]->value;
  if (parent0.type.variable_type != graph::VariableType::COL_SIMPLEX_MATRIX or
      parent0.type.cols != 1) {
    throw std::invalid_argument(
        "Categorical parent must be a one-column simplex");
  }
}

graph::natural_t Categorical::_natural_sampler(std::mt19937& gen) const {
  const Eigen::MatrixXd& matrix = in_nodes[0]->value._matrix;
  assert(matrix.cols() == 1);

  // distrib(c0.begin(), c0.end()) fails on CircleCI saying that there are no
  // such methods on a Block<>; we make this seemingly unnecessary copy to
  // a vector to get around that.

  const auto& c0 = matrix.col(0);
  std::vector<double> v;
  for (int i = 0; i < c0.size(); i += 1) {
    v.push_back(c0(i));
  }
  std::discrete_distribution<> distrib(v.begin(), v.end());
  return (graph::natural_t)distrib(gen);
}

double Categorical::log_prob(const graph::NodeValue& value) const {
  assert(in_nodes.size() == 1);
  assert(in_nodes[0] != 0);
  const Eigen::MatrixXd& matrix = in_nodes[0]->value._matrix;
  double prob = 0.0;
  graph::natural_t r = (graph::natural_t)matrix.rows();
  if (0 <= value._natural and value._natural < r) {
    prob = matrix(value._natural, 0);
  }
  return std::log(prob);
}

void Categorical::log_prob_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& log_probs) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  log_probs = Eigen::MatrixXd(value._nmatrix.rows(), value._nmatrix.cols());
  uint rows = static_cast<uint>(value._nmatrix.rows());
  uint cols = static_cast<uint>(value._nmatrix.cols());
  for (uint r = 0; r < rows; r += 1) {
    for (uint c = 0; c < cols; c += 1) {
      log_probs(r, c) = log_prob(graph::NodeValue(value._nmatrix(r, c)));
    }
  }
}

// The likelihood L(x|p_0, ... p_(k-1)) (where x is the outcome and the p's are
// the k parameters of the categorical distribution) is p_0 if x is 0, p_1 if
// x is 1, and so on.
//
// We need the gradient of log(L(x|p_0...)); how do we get a gradient on
// a discrete function?
//
// TODO: Yeah, how?

void Categorical::gradient_log_prob_value(
    const graph::NodeValue& /* value */,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  grad1 += 0; // TODO
  grad2 += 0; // TODO
}

void Categorical::gradient_log_prob_param(
    const graph::NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  grad1 += 0; // TODO
  grad2 += 0; // TODO
}

void Categorical::backward_param(const graph::NodeValue& value, double adjunct)
    const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1._double += 0; // TODO
  }
}

void Categorical::backward_param_iid(const graph::NodeValue& value) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1._double += 0; // TODO
  }
}

void Categorical::backward_param_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  if (in_nodes[0]->needs_gradient()) {
    in_nodes[0]->back_grad1._double += 0; // TODO
  }
}

} // namespace distribution
} // namespace beanmachine
