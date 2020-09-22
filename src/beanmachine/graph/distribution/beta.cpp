// Copyright (c) Facebook, Inc. and its affiliates.
#include <cmath>

#include "beanmachine/graph/distribution/beta.h"
#include "beanmachine/graph/util.h"

namespace beanmachine {
namespace distribution {

Beta::Beta(
    graph::AtomicType sample_type,
    const std::vector<graph::Node*>& in_nodes)
    : Distribution(graph::DistributionType::BETA, sample_type) {
  // a Beta has two parents which are real numbers and it outputs a probability
  if (sample_type != graph::AtomicType::PROBABILITY) {
    throw std::invalid_argument("Beta produces probability samples");
  }
  if (in_nodes.size() != 2) {
    throw std::invalid_argument(
        "Beta distribution must have exactly two parents");
  }
  if (in_nodes[0]->value.type != graph::AtomicType::POS_REAL or
      in_nodes[1]->value.type != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument("Beta parents must be positive real-valued");
  }
}

double Beta::_double_sampler(std::mt19937& gen) const {
  return util::sample_beta(
      gen, in_nodes[0]->value._double, in_nodes[1]->value._double);
}

double Beta::log_prob(const graph::AtomicValue& value) const {
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  double ret_val = 0.0;
  auto update_logprob = [&] (double val)
  {
    ret_val += (param_a - 1) * log(val) + (param_b - 1) * log(1 - val);
  };
  if (value.type.variable_type == graph::VariableType::SCALAR) {
    update_logprob(value._double);
    ret_val += lgamma(param_a + param_b) - lgamma(param_a) - lgamma(param_b);
    return ret_val;
  }

  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  assert(value.type.rows * value.type.cols > 1);
  const uint size = value._matrix.size();
  for (uint i = 0; i < size; i++) {
    update_logprob(*(value._matrix.data() + i));
  }
  ret_val += size * (lgamma(param_a + param_b) - lgamma(param_a) - lgamma(param_b));
  return ret_val;
}

// Note log_prob(x | a, b) = (a-1) log(x) + (b-1) log(1-x) + log G(a+b) - log
// G(a) - log G(b) grad1 w.r.t. x =  (a-1) / x - (b-1) / (1-x) grad2 w.r.t. x =
// - (a-1) / x^2 - (b-1) / (1-x)^2 grad1 w.r.t. params = (log(x) + digamma(a+b)
// - digamma(a)) a'
//                        + (log(1-x) + digamma(a+b) - digamma(b)) b'
// grad2 w.r.t. params =
//   (polygamma(1, a+b) - polygamma(1, a)) a'^2 + (log(x) + digamma(a+b) -
//   digamma(a)) a'' (polygamma(1, a+b) - polygamma(1, b)) b'^2 + (log(1-x) +
//   digamma(a+b) - digamma(b)) b''
void _gradient_log_prob_value(
    const double& val,
    double& grad1,
    double& grad2,
    const double& param_a,
    const double& param_b) {
  grad1 += (param_a - 1) / val - (param_b - 1) / (1 - val);
  grad2 +=
      -(param_a - 1) / (val * val) - (param_b - 1) / ((1 - val) * (1 - val));
}

void Beta::gradient_log_prob_value(
    const graph::AtomicValue& value,
    double& grad1,
    double& grad2) const {
  assert(value.type.variable_type == graph::VariableType::SCALAR);
  _gradient_log_prob_value(
      value._double,
      grad1,
      grad2,
      in_nodes[0]->value._double,
      in_nodes[1]->value._double);
}

void Beta::gradient_log_prob_value(
    const graph::AtomicValue& value,
    Eigen::MatrixXd& grad1,
    Eigen::MatrixXd& grad2_diag) const {
  assert(value.type.variable_type == graph::VariableType::BROADCAST_MATRIX);
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  uint size = value._matrix.size();
  assert(size == grad1.size() and size == grad2_diag.size());
  for (uint i = 0; i < size; i++) {
    _gradient_log_prob_value(
        *(value._matrix.data() + i),
        *(grad1.data() + i),
        *(grad2_diag.data() + i),
        param_a,
        param_b);
  }
}

void Beta::gradient_log_prob_param(
    const graph::AtomicValue& value,
    double& grad1,
    double& grad2) const {
  Eigen::Matrix<double, 1, 2> jacobian;
  Eigen::Matrix2d hessian;
  Eigen::MatrixXd pseudo_input;
  compute_jacobian_hessian(value, jacobian, hessian);
  gradient_propagation_scalar_to_scalar(
      true, jacobian, hessian, grad1, grad2, pseudo_input, pseudo_input);
}

void Beta::compute_jacobian_hessian(
    const graph::AtomicValue& value,
    Eigen::Matrix<double, 1, 2>& jacobian,
    Eigen::Matrix2d& hessian) const {
  double param_a = in_nodes[0]->value._double;
  double param_b = in_nodes[1]->value._double;
  double digamma_a_p_b = util::polygamma(0, param_a + param_b); // digamma(a+b)
  double digamma_diff_a = digamma_a_p_b - util::polygamma(0, param_a);
  double digamma_diff_b = digamma_a_p_b - util::polygamma(0, param_b);
  double poly1_a_p_b =
      util::polygamma(1, param_a + param_b); // polygamma(1, a+b)

  *hessian.data() = poly1_a_p_b - util::polygamma(1, param_a);
  *(hessian.data() + 1) = *(hessian.data() + 2) = poly1_a_p_b;
  *(hessian.data() + 3) = poly1_a_p_b - util::polygamma(1, param_b);

  if (value.type.variable_type == graph::VariableType::SCALAR) {
    *jacobian.data() = std::log(value._double) + digamma_diff_a;
    *(jacobian.data() + 1) = std::log(1 - value._double) + digamma_diff_b;
    return;
  }

  uint size = value._matrix.size();
  assert(size > 1);
  *jacobian.data() = size * digamma_diff_a;
  *(jacobian.data() + 1) = size * digamma_diff_b;
  for (uint i = 0; i < size; i++) {
    *jacobian.data() += std::log(*(value._matrix.data() + i));
    *(jacobian.data() + 1) += std::log(1 - *(value._matrix.data() + i));
  }
  hessian *= size;
}

} // namespace distribution
} // namespace beanmachine
