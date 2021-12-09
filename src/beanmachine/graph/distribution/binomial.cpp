/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <unsupported/Eigen/SpecialFunctions>

#include "beanmachine/graph/distribution/binomial.h"

namespace beanmachine {
namespace distribution {

Binomial::Binomial(
    graph::AtomicType sample_type,
    const std::vector<graph::Node*>& in_nodes)
    : Distribution(graph::DistributionType::BINOMIAL, sample_type) {
  // a Binomial has two parents -- natural, probability and outputs a natural
  if (sample_type != graph::AtomicType::NATURAL) {
    throw std::invalid_argument("Binomial produces natural number samples");
  }
  if (in_nodes.size() != 2) {
    throw std::invalid_argument(
        "Binomial distribution must have exactly two parents");
  }
  if (in_nodes[0]->value.type != graph::AtomicType::NATURAL or
      in_nodes[1]->value.type != graph::AtomicType::PROBABILITY) {
    throw std::invalid_argument(
        "Binomial parents must be a natural number and a probability");
  }
}

graph::natural_t Binomial::_natural_sampler(std::mt19937& gen) const {
  graph::natural_t param_n = in_nodes[0]->value._natural;
  double param_p = in_nodes[1]->value._double;
  std::binomial_distribution<graph::natural_t> distrib(param_n, param_p);
  return distrib(gen);
}

double Binomial::log_prob(const graph::NodeValue& value) const {
  graph::natural_t n = in_nodes[0]->value._natural;
  double p = in_nodes[1]->value._double;
  double ret_val = 0;
  if (value.type.variable_type == graph::VariableType::SCALAR) {
    graph::natural_t k = value._natural;
    if (k > n) {
      return -std::numeric_limits<double>::infinity();
    }
    // we will try not to evaluate log(p) or log(1-p) unless needed
    if (k > 0) {
      ret_val += k * log(p);
    }
    if (k < n) {
      ret_val += (n - k) * log(1 - p);
    }
    // note: Gamma(n+1) = n!
    ret_val += std::lgamma(n + 1) - std::lgamma(k + 1) - std::lgamma(n - k + 1);
  } else if (
      value.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    if ((value._nmatrix.array() > n).any()) {
      return -std::numeric_limits<double>::infinity();
    }
    int size = static_cast<int>(value._nmatrix.size());
    double sum_k = static_cast<double>(value._nmatrix.sum());

    // we will try not to evaluate log(p) or log(1-p) unless needed
    if ((value._nmatrix.array() > 0).any()) {
      ret_val += sum_k * log(p);
    }
    if ((value._nmatrix.array() < n).any()) {
      ret_val += (n * size - sum_k) * log(1 - p);
    }

    // note: Gamma(n+1) = n!
    Eigen::MatrixXd value_double = value._nmatrix.cast<double>();
    double k_factorial_sum = (value_double.array() + 1).lgamma().sum();
    double n_k_factorial_sum = (n - value_double.array() + 1).lgamma().sum();
    ret_val += std::lgamma(n + 1) * size - k_factorial_sum - n_k_factorial_sum;
  }
  return ret_val;
}

void Binomial::log_prob_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& log_probs) const {
  graph::natural_t n = in_nodes[0]->value._natural;
  double p = in_nodes[1]->value._double;
  Eigen::MatrixXd value_double = value._nmatrix.cast<double>();
  log_probs = value_double.array() * log(p) +
      (n - value_double.array()) * log(1 - p) + std::lgamma(n + 1) -
      (value_double.array() + 1).lgamma() -
      (n - value_double.array() + 1).lgamma();
}

// log_prob is k log(p) + (n-k) log(1-p) as a function of k
// grad1 is  (log(p) - log(1-p))
// grad2 is  0
void Binomial::gradient_log_prob_value(
    const graph::NodeValue& /* value */,
    double& grad1,
    double& /* grad2 */) const {
  // nothing to do here since the value is a natural number and we can't
  // compute gradients w.r.t. naturals
  double p = in_nodes[1]->value._double;
  grad1 += std::log(p) - std::log(1 - p);
  // grad2 += 0;
}

// log_prob is k log(p) + (n-k) log(1-p) as a function of p
// grad1 is  (k/p) * p' - ((n-k) / (1-p)) * p'
// grad2 is -(k/p^2) * p'^2 + (k/p) * p'' - ((n-k)/(1-p)^2) * p'^2 -
// ((n-k)/(1-p)) * p''
void Binomial::gradient_log_prob_param(
    const graph::NodeValue& value,
    double& grad1,
    double& grad2) const {
  double n = (double)in_nodes[0]->value._natural;
  double p = in_nodes[1]->value._double;
  double k = (double)value._natural;
  // first compute gradients w.r.t. p
  double grad_p = (k / p) - (n - k) / (1 - p);
  double grad2_p2 = (-k / (p * p)) - (n - k) / ((1 - p) * (1 - p));
  grad1 += grad_p * in_nodes[1]->grad1;
  grad2 += grad2_p2 * in_nodes[1]->grad1 * in_nodes[1]->grad1 +
      grad_p * in_nodes[1]->grad2;
}

// log_prob is k log(p) + (n-k) log(1-p) as a function of p
// grad1 is  (k/p) * p' - ((n-k) / (1-p)) * p'
void Binomial::backward_param(const graph::NodeValue& value, double adjunct)
    const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  double n = (double)in_nodes[0]->value._natural;
  double p = in_nodes[1]->value._double;
  double k = (double)value._natural;

  if (in_nodes[1]->needs_gradient()) {
    double grad = k / p - (n - k) / (1 - p);
    in_nodes[1]->back_grad1._double += adjunct * grad;
  }
}

void Binomial::backward_param_iid(const graph::NodeValue& value) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);

  if (in_nodes[1]->needs_gradient()) {
    double n = (double)in_nodes[0]->value._natural;
    double p = in_nodes[1]->value._double;
    int size = static_cast<int>(value._nmatrix.size());
    double sum_k = static_cast<double>(value._nmatrix.sum());
    double grad = sum_k / p - (size * n - sum_k) / (1 - p);
    in_nodes[1]->back_grad1._double += grad;
  }
}

void Binomial::backward_param_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);

  if (in_nodes[1]->needs_gradient()) {
    double n = (double)in_nodes[0]->value._natural;
    double p = in_nodes[1]->value._double;

    double sum_adjunct = adjunct.sum();
    double sum_k_adjunct =
        (value._nmatrix.cast<double>().array() * adjunct.array()).sum();
    double grad =
        sum_k_adjunct / p - (sum_adjunct * n - sum_k_adjunct) / (1 - p);
    in_nodes[1]->back_grad1._double += grad;
  }
}

} // namespace distribution
} // namespace beanmachine
