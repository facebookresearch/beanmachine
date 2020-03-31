// Copyright (c) Facebook, Inc. and its affiliates.
#include <string>
#include <random>

#include "beanmachine/graph/distribution/normal.h"

namespace beanmachine {
namespace distribution {

using namespace graph;

const double PI = 3.141592653589793;

Normal::Normal(
    AtomicType sample_type,
    const std::vector<Node*>& in_nodes)
    : Distribution(DistributionType::NORMAL, sample_type) {
  // a Normal distribution has two parents
  // mean -> real, sigma -> positive real
  if (in_nodes.size() != 2) {
    throw std::invalid_argument(
        "Normal distribution must have exactly two parents");
  }
  if (in_nodes[0]->value.type != graph::AtomicType::REAL
      or in_nodes[1]->value.type != graph::AtomicType::POS_REAL) {
    throw std::invalid_argument(
        "Normal parents must be a real number and a positive real number");
  }
  // only real-valued samples are possible
  if (sample_type != AtomicType::REAL) {
    throw std::invalid_argument("Normal distribution produces real number samples");
  }
}

AtomicValue Normal::sample(std::mt19937& gen) const {
  std::normal_distribution<double> dist(
    in_nodes[0]->value._double, in_nodes[1]->value._double);
  return AtomicValue(dist(gen));
}

// log_prob of a normal: - log(s) -0.5 log(2*pi) - 0.5 (x - m)^2 / s^2
// grad  w.r.t. value x: - (x - m) / s^2
// grad2 w.r.t. value x: - 1 / s^2
// grad  w.r.t. s : -1/s + (x-m)^2 / s^3
// grad2 w.r.t. s : 1/s^2 - 3 (x-m)^2 / s^4
// grad  w.r.t. m : (x - m) / s^2
// grad2 w.r.t. m : -1 / s^2
double Normal::log_prob(const AtomicValue& value) const {
  double x = value._double;
  double m = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  return -std::log(s) - 0.5 * std::log(2 * PI) - 0.5 * (x - m) * (x - m) / (s * s);
}

void Normal::gradient_log_prob_value(
    const AtomicValue& value, double& grad1, double& grad2) const {
  double x = value._double;
  double m = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  double s_sq = s * s;
  grad1 += - (x - m) / s_sq;
  grad2 += - 1 / s_sq;
}

void Normal::gradient_log_prob_param(
    const AtomicValue& value, double& grad1, double& grad2) const {
  double x = value._double;
  double m = in_nodes[0]->value._double;
  double s = in_nodes[1]->value._double;
  double s_sq = s * s;
  // gradients of m should be non-zero before computing gradients w.r.t. m
  double m_grad = in_nodes[0]->grad1;
  double m_grad2 = in_nodes[0]->grad2;
  if (m_grad != 0 or m_grad2 != 0) {
    double grad_m = (x - m) / s_sq;
    double grad2_m2 = -1 / s_sq;
    grad1 += grad_m * m_grad;
    grad2 += grad2_m2 * m_grad * m_grad + grad_m * m_grad2;
  }
  double s_grad = in_nodes[1]->grad1;
  double s_grad2 = in_nodes[1]->grad2;
  if (s_grad != 0 or s_grad2 != 0) {
    double grad_s = -1 / s + (x-m) * (x-m) / (s * s * s);
    double grad2_s2 = 1 / s_sq - 3 * (x-m)*(x-m) / (s_sq * s_sq);
    grad1 += grad_s * s_grad;
    grad2 += grad2_s2 * s_grad * s_grad + grad_s * s_grad2;
  }
}

} // namespace distribution
} // namespace beanmachine
