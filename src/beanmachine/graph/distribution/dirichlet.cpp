/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include "beanmachine/graph/distribution/dirichlet.h"
#include "beanmachine/graph/util.h"

using namespace torch::indexing;

namespace beanmachine {
namespace distribution {


Dirichlet::Dirichlet(
    graph::ValueType sample_type,
    const std::vector<graph::Node*>& in_nodes)
    : Distribution(graph::DistributionType::DIRICHLET, sample_type) {
  // a Dirichlet has one parent which is a positive real matrix
  // and outputs a col simplex matrix
  if (sample_type.atomic_type != graph::AtomicType::PROBABILITY) {
    throw std::invalid_argument("Dirichlet produces probability samples");
  }
  if (sample_type.variable_type != graph::VariableType::COL_SIMPLEX_MATRIX) {
    throw std::invalid_argument(
        "Dirichlet produces COL_SIMPLEX_MATRIX samples");
  }
  if (in_nodes.size() != 1) {
    throw std::invalid_argument(
        "Dirichlet distribution must have exactly one parent");
  }
  if (in_nodes[0]->value.type.variable_type !=
          graph::VariableType::BROADCAST_MATRIX or
      in_nodes[0]->value.type.atomic_type != graph::AtomicType::POS_REAL or
      in_nodes[0]->value.type.cols != 1) {
    throw std::invalid_argument(
        "Dirichlet parent must be a positive real-valued matrix with one column.");
  }
}

torch::Tensor Dirichlet::_matrix_sampler(std::mt19937& gen) const {
  int n_rows = static_cast<int>(in_nodes[0]->value._value.size(0));
  torch::Tensor sample = torch::empty({n_rows, 1});

  torch::Tensor param = in_nodes[0]->value._value;
  for (int i = 0; i < n_rows; i++) {
    std::gamma_distribution<double> gamma_dist(param.index({i}).item().toDouble(), 1);
    sample.index_put_({i}, gamma_dist(gen));
  }
  return sample / sample.sum().item().toDouble();
}

double Dirichlet::log_prob(const graph::NodeValue& value) const {
  assert(value.type.variable_type == graph::VariableType::COL_SIMPLEX_MATRIX);
  assert(value.type.cols == 1);
  torch::Tensor param = in_nodes[0]->value._value;

  double log_prob = 0.0;
  for (int i = 0; i < param.size(0); i++) {
    double alpha = param.index({i,0}).item().toDouble();
    log_prob -= lgamma(alpha);
    log_prob += std::log(value._value.index({i,0}).item().toDouble()) * (alpha - 1);
  }
  log_prob += lgamma(param.sum(0)).item().toDouble();

  return log_prob;
}

void Dirichlet::log_prob_iid(
    const graph::NodeValue& /* value */,
    torch::Tensor& /* log_probs */) const {}

// log_prob(x | a) =
// log G(\sum(a_i)) - \sum(log G(a_i)) + \sum(log(x_i) * (a_i - 1))
// grad1 w.r.t. x_i = (a_i - 1) / x_i
// grad1 w.r.t. a_i = digamma(\sum(a_i)) - digamma(a_i) + log(x_i)
// First order chain rule: f(g(x))' = f'(g(x)) g'(x)
void Dirichlet::backward_value(
    const graph::NodeValue& value,
    graph::DoubleMatrix& back_grad,
    double adjunct) const {
  assert(value.type.variable_type == graph::VariableType::COL_SIMPLEX_MATRIX);
  torch::Tensor x = value._value;
  torch::Tensor param = in_nodes[0]->value._value;
  for (int i = 0; i < param.size(0); i++) {
    back_grad._value.index_put_(
      {i,Slice()},
      back_grad._value.index({i,Slice()})
      + adjunct * (param.index({i,Slice()}) - 1) / x.index({i,Slice()})
    );
  }
}

void Dirichlet::backward_param(const graph::NodeValue& value, double adjunct)
    const {
  assert(value.type.variable_type == graph::VariableType::COL_SIMPLEX_MATRIX);
  torch::Tensor x = value._value;
  torch::Tensor param = in_nodes[0]->value._value;
  double digamma_sum = util::polygamma(0, param.sum().item().toDouble());
  if (in_nodes[0]->needs_gradient()) {
    for (int i = 0; i < param.size(0); i++) {
      double jacob =
          std::log(x.index({i,0}).item().toDouble()) + digamma_sum - util::polygamma(0, param.index({i,0}).item().toDouble());
      in_nodes[0]->back_grad1._value.index_put_(
        {i,Slice()},
        in_nodes[0]->back_grad1._value.index({i,Slice()}) + adjunct * jacob
      );
    }
  }
}

} // namespace distribution
} // namespace beanmachine
