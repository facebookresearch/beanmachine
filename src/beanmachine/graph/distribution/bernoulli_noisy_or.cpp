/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include "beanmachine/graph/distribution/bernoulli_noisy_or.h"

namespace beanmachine {
namespace distribution {

// helper function to compute log(1 - exp(-x)) for x >= 0
static inline double log1mexpm(double x) {
  // see the following document for an explanation of why we switch at .69315
  // https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
  // essentially for small values expm1 prevents underflow to 0.0 and for
  // large values it prevents overflow to 1.0.
  //   log1mexp(1e-20) -> -46.051701859880914
  //   log1mexp(40) -> -4.248354255291589e-18
  if (x < 0.69315) {
    return std::log(-std::expm1(-x));
  } else {
    return std::log1p(-std::exp(-x));
  }
}

BernoulliNoisyOr::BernoulliNoisyOr(
    graph::AtomicType sample_type,
    const std::vector<graph::Node*>& in_nodes)
    : Distribution(graph::DistributionType::BERNOULLI_NOISY_OR, sample_type) {
  if (sample_type != graph::AtomicType::BOOLEAN) {
    throw std::invalid_argument(
        "BernoulliNoisyOr produces boolean valued samples");
  }
  // a BernoulliNoisyOr can only have one parent which must be a POS_REAL (>=0)
  if (in_nodes.size() != 1) {
    throw std::invalid_argument(
        "BernoulliNoisyOr distribution must have exactly one parent");
  }
  const auto& parent0 = in_nodes[0]->value;
  if (parent0.type != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "BernoulliNoisyOr parent probability must be positive real-valued");
  }
}

bool BernoulliNoisyOr::_bool_sampler(std::mt19937& gen) const {
  double param = in_nodes[0]->value._double;
  double prob = 1 - exp(-param);
  std::bernoulli_distribution distrib(prob);
  return (bool)distrib(gen);
}

double BernoulliNoisyOr::log_prob(const graph::NodeValue& value) const {
  double param = in_nodes[0]->value._double;
  int size;
  double sum_x, result;

  if (value.type.variable_type == graph::VariableType::SCALAR) {
    size = 1;
    sum_x = (double)value._bool;
  } else if (
      value.type.variable_type == graph::VariableType::BROADCAST_MATRIX) {
    size = static_cast<int>(value._bmatrix.size());
    sum_x = static_cast<int>(value._bmatrix.count());
  } else {
    throw std::runtime_error(
        "Normal::log_prob applied to invalid variable type");
  }
  result = sum_x * log1mexpm(param) + (size - sum_x) * (-param);
  return result;
}

void BernoulliNoisyOr::log_prob_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& log_probs) const {
  double param = in_nodes[0]->value._double;
  double logterm = log1mexpm(param);
  Eigen::MatrixXd value_double = value._bmatrix.cast<double>();
  log_probs =
      value_double.array() * logterm + (1 - value_double.array()) * (-param);
}

// x ~ BernoulliNoisyOr(y) = Bernoulli(1 - exp(-y))
// f(x, y) = x log(1 - exp(-y)) + (1-x)(-y)
// w.r.t. x:   f' = log(1 - exp(-y)) + y     f'' = 0
// w.r.t. y:   f' =  [x exp(-y) / (1 - exp(-y)) - (1-x)] y' = [x / (1 - exp(-y))
// - 1] y'
//             f'' = [- x exp(-y) /(1 - exp(-y))^2] y'^2 + [x / (1 - exp(-y)) -
//             1] y''
void BernoulliNoisyOr::gradient_log_prob_value(
    const graph::NodeValue& /* value */,
    double& grad1,
    double& /* grad2 */) const {
  double param = in_nodes[0]->value._double;
  grad1 += log1mexpm(param) + param;
  // grad2 += 0
}

void BernoulliNoisyOr::gradient_log_prob_param(
    const graph::NodeValue& value,
    double& grad1,
    double& grad2) const {
  double param = in_nodes[0]->value._double;
  double mexpm1m = -std::expm1(-param); // 1 - exp(-param)
  double val = (double)value._bool;
  double grad_param = val / mexpm1m - 1;
  double grad2_param = -val * std::exp(-param) / (mexpm1m * mexpm1m);
  grad1 += grad_param * in_nodes[0]->grad1;
  grad2 += grad2_param * in_nodes[0]->grad1 * in_nodes[0]->grad1 +
      grad_param * in_nodes[0]->grad2;
}

void BernoulliNoisyOr::backward_param(
    const graph::NodeValue& value,
    double adjunct) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  if (in_nodes[0]->needs_gradient()) {
    double param = in_nodes[0]->value._double;
    double mexpm1m = -std::expm1(-param); // 1 - exp(-param)
    double val = (double)value._bool;
    double grad_param = val / mexpm1m - 1;
    in_nodes[0]->back_grad1._double += adjunct * grad_param;
  }
}

void BernoulliNoisyOr::backward_param_iid(const graph::NodeValue& value) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  if (in_nodes[0]->needs_gradient()) {
    double param = in_nodes[0]->value._double;
    double mexpm1m = -std::expm1(-param); // 1 - exp(-param)
    double val_sum = (double)value._bmatrix.count();
    int size = static_cast<int>(value._bmatrix.size());
    in_nodes[0]->back_grad1._double += val_sum / mexpm1m - size;
  }
}

void BernoulliNoisyOr::backward_param_iid(
    const graph::NodeValue& value,
    Eigen::MatrixXd& adjunct) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  if (in_nodes[0]->needs_gradient()) {
    double param = in_nodes[0]->value._double;
    double mexpm1m = -std::expm1(-param); // 1 - exp(-param)
    double sum_adjunct = adjunct.sum();
    double sum_x_adjunct =
        (value._bmatrix.cast<double>().array() * adjunct.array()).sum();
    in_nodes[0]->back_grad1._double += sum_x_adjunct / mexpm1m - sum_adjunct;
  }
}

} // namespace distribution
} // namespace beanmachine
