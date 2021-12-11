/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <random>
#include <string>

#include "beanmachine/graph/distribution/bernoulli_logit.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace distribution {

using namespace graph;

BernoulliLogit::BernoulliLogit(
    AtomicType sample_type,
    const std::vector<Node*>& in_nodes)
    : Distribution(DistributionType::BERNOULLI_LOGIT, sample_type) {
  // a BernoulliLogit distribution has one parent which is a real value
  if (in_nodes.size() != 1) {
    throw std::invalid_argument(
        "BernoulliLogit distribution must have exactly one parent");
  }
  if (in_nodes[0]->value.type != graph::AtomicType::REAL) {
    throw std::invalid_argument("BernoulliLogit parent must be a real value");
  }
  // only BOOLEAN-valued samples are possible
  if (sample_type != AtomicType::BOOLEAN) {
    throw std::invalid_argument(
        "BernoulliLogit distribution produces boolean samples");
  }
}

bool BernoulliLogit::_bool_sampler(std::mt19937& gen) const {
  double logodds = in_nodes[0]->value._double;
  return (bool)util::sample_logodds(gen, logodds);
}

// log_prob of a BernoulliLogit with parameter l (logodds)
// f(x, l) = - x log(1 + exp(-l)) - (1-x) log(1 + exp(l))
// df / dx = - log(1 + exp(-l)) + log(1 + exp(l))
//         = l
// d2f / dx2 = 0
// df / dl = x exp(-l) / (1 + exp(-l)) - (1-x) exp(l) / (1 + exp(l))
//          = x /(1 + exp(l)) - (1 - x) / (1 + exp(-l))
// d2f dl2 = - x exp(l) / (1 + exp(l))^2 - (1-x) exp(-l) / (1 + exp(-l))^2
//         = -1 /[(1 + exp(-l)) (1 + exp(l))]
//         = -1 / (2 + exp(-l) + exp(l))
// First order chain rule: f(g(x))' = f'(g(x)) g'(x),
// - In backward propagation, f'(g(x)) is given by adjunct, the above equation
// computes g'(x). [g is the current function f is the final target]
// - In forward propagation, g'(x) is given by in_nodes[x]->grad1,
// the above equation computes f'(g) [f is the current function g is the input]
double BernoulliLogit::log_prob(const NodeValue& value) const {
  double l = in_nodes[0]->value._double;

  if (value.type.variable_type == graph::VariableType::SCALAR) {
    return value._bool ? -util::log1pexp(-l) : -util::log1pexp(l);
  } else if (
      value.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    int size = static_cast<int>(value._bmatrix.size());
    int n_positive = static_cast<int>(value._bmatrix.count());
    return -util::log1pexp(-l) * n_positive -
        util::log1pexp(l) * (size - n_positive);
  } else {
    throw std::runtime_error(
        "BernoulliLogit::log_prob applied to invalid variable type");
  }
}

void BernoulliLogit::log_prob_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& log_probs) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double l = in_nodes[0]->value._double;
  double pos_val = -util::log1pexp(-l);
  double neg_val = -util::log1pexp(l);
  log_probs = Eigen::MatrixXd::Constant(
      value._bmatrix.rows(), value._bmatrix.cols(), neg_val);
  log_probs = value._bmatrix.select(pos_val, log_probs);
}

void BernoulliLogit::gradient_log_prob_value(
    const NodeValue& /* value */,
    double& grad1,
    double& /* grad2 */) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double l = in_nodes[0]->value._double;
  grad1 += l;
  // grad2 += 0
}

double BernoulliLogit::_grad1_log_prob_param(bool x, double l) {
  return x ? 1 / (1 + std::exp(l)) : -1 / (1 + std::exp(-l));
}

void BernoulliLogit::gradient_log_prob_param(
    const NodeValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  bool x = value._bool;
  double l = in_nodes[0]->value._double;
  // We will compute the gradients w.r.t. each the parameter only if
  // the gradients of the parameter w.r.t. the source variable is non-zero
  double l_grad = in_nodes[0]->grad1;
  double l_grad2 = in_nodes[0]->grad2;
  if (l_grad != 0 or l_grad2 != 0) {
    double grad_l = _grad1_log_prob_param(x, l);
    double grad2_l2 = -1 / (2 + std::exp(-l) + std::exp(l));
    grad1 += grad_l * l_grad;
    grad2 += grad2_l2 * l_grad * l_grad + grad_l * l_grad2;
  }
}

void BernoulliLogit::backward_param(
    const graph::NodeValue& value,
    double adjunct) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  if (in_nodes[0]->needs_gradient()) {
    bool x = value._bool;
    double l = in_nodes[0]->value._double;
    in_nodes[0]->back_grad1._double += adjunct * _grad1_log_prob_param(x, l);
  }
}

void BernoulliLogit::backward_param_iid(const graph::NodeValue& value) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  if (in_nodes[0]->needs_gradient()) {
    double l = in_nodes[0]->value._double;
    int size = static_cast<int>(value._bmatrix.size());
    int n_positive = static_cast<int>(value._bmatrix.count());
    in_nodes[0]->back_grad1._double +=
        (1 / (1 + std::exp(l)) * n_positive -
         1 / (1 + std::exp(-l)) * (size - n_positive));
  }
}

void BernoulliLogit::backward_param_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  if (in_nodes[0]->needs_gradient()) {
    double l = in_nodes[0]->value._double;
    double sum_adjunct = adjunct.sum();
    double sum_pos_adjunct =
        (value._bmatrix.cast<double>().array() * adjunct.array()).sum();
    in_nodes[0]->back_grad1._double +=
        (1 / (1 + std::exp(l)) * sum_pos_adjunct -
         1 / (1 + std::exp(-l)) * (sum_adjunct - sum_pos_adjunct));
  }
}

} // namespace distribution
} // namespace beanmachine
